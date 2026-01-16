#ifndef ZNS_ARC_SEQLOCK_H
#define ZNS_ARC_SEQLOCK_H

#include <stddef.h>      // size_t
#include <stdint.h>      // uint64_t, etc
#include <stdbool.h>     // bool
#include <stdatomic.h>   // atomic types

// ============================================================
// ZNS SSD 환경 설정
// ============================================================
#define PAGE_SIZE 4096
#define ZNS_ZONE_SIZE (256 * 1024 * 1024)
#define PAGES_PER_ZONE (ZNS_ZONE_SIZE / PAGE_SIZE)
#define TOTAL_WORKSET_PAGES 400000
#define CACHE_SIZE_RATIO 0.1
#define CACHE_SIZE ((int)(TOTAL_WORKSET_PAGES * CACHE_SIZE_RATIO))
#define ZONE_BATCH_SIZE 4
#define ACTIVE_ZONES_LIMIT 14
#define GHOST_HIT_WINDOW 10

// 해시테이블 설정
#define HASH_INITIAL_CAPACITY 256
#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

// 메모리 풀 설정
#define POOL_SIZE 600000
#define POOL_FALLBACK_THRESHOLD 0.95

// Spinlock 설정
#define SPINLOCK_TIMEOUT_NS 100000 // 100μs

// Deferred Eviction Queue 설정
#define DEFERRED_QUEUE_SIZE 64




// ============================================================
// Seqlock 구조체 - 원자적 피처 읽기를 위한 경량 동기화
// ============================================================
typedef struct {
    atomic_uint sequence;
} seqlock_t;

// Seqlock 인라인 함수들
static inline void seqlock_init(seqlock_t *lock) {
    atomic_init(&lock->sequence, 0);
}

static inline void seqlock_write_begin(seqlock_t *lock) {
    uint32_t seq = atomic_load_explicit(&lock->sequence, memory_order_relaxed);
    atomic_store_explicit(&lock->sequence, seq + 1, memory_order_release);
}

static inline void seqlock_write_end(seqlock_t *lock) {
    uint32_t seq = atomic_load_explicit(&lock->sequence, memory_order_relaxed);
    atomic_store_explicit(&lock->sequence, seq + 1, memory_order_release);
}

static inline uint32_t seqlock_read_begin(seqlock_t *lock) {
    uint32_t seq;
    do {
        seq = atomic_load_explicit(&lock->sequence, memory_order_acquire);
    } while (seq & 1); // 홀수면 쓰기 중이므로 대기
    return seq;
}

static inline int seqlock_read_retry(seqlock_t *lock, uint32_t seq) {
    atomic_thread_fence(memory_order_acquire);
    return atomic_load_explicit(&lock->sequence, memory_order_relaxed) != seq;
}

// ============================================================
// 데이터 구조체
// ============================================================

// Zone 정보
typedef struct {
    int zone_id;
    int valid_page_count;
    int total_page_count;
    bool is_active;
    long last_access_tick;
    atomic_flag zone_lock;
} ZoneInfo;

// Page 특징 - Seqlock으로 보호됨
typedef struct {
    seqlock_t lock;  // 원자적 읽기를 위한 seqlock
    int access_count;
    long last_access_tick;
    int list_type;
    long last_reuse_distance;
    float avg_reuse_interval;
    float reuse_variance;
    int reuse_count;
    long* reuse_history;
    int history_size;
    float neighbor_avg_access;
    float neighbor_avg_idle;
} PageFeatures;

// Page 엔트리
typedef struct PageEntry {
    int page_id;
    int zone_id;
    PageFeatures feat;
    struct PageEntry* prev;
    struct PageEntry* next;
    bool in_pool;
} PageEntry;

// 해시테이블
typedef struct {
    int page_id;
    PageEntry* entry;
} HashEntry;

typedef struct {
    HashEntry* entries;
    size_t capacity;
    size_t length;
} HashTable;

// LRU 리스트
typedef struct {
    PageEntry* head;
    PageEntry* tail;
    int size;
    HashTable* hash;
} LRUList;

// 메모리 풀
typedef struct {
    PageEntry* pool;
    atomic_uint* free_bitmap;
    atomic_int pool_size;
    int capacity;
} PageEntryPool;

// Deferred Eviction Batch
typedef struct {
    PageEntry* pages[ZONE_BATCH_SIZE];
    int count;
    int target_zone;
    int source_type; // 1=T1, 2=T2
    long timestamp;
} DeferredBatch;

// Deferred Eviction Queue
typedef struct {
    DeferredBatch queue[DEFERRED_QUEUE_SIZE];
    atomic_int head;
    atomic_int tail;
    atomic_int size;
} DeferredEvictionQueue;

// 성능 모니터
typedef struct {
    atomic_long total_spins;
    atomic_long timeout_count;
    atomic_long max_spin_ns;
    atomic_long pool_fallback_count;
    atomic_long zone_rotation_count;
    atomic_long deferred_eviction_count;
    atomic_long deferred_processed_count;
    atomic_long zone_contention[1024];
    atomic_long seqlock_retry_count;  // Seqlock 재시도 횟수
} PerformanceMonitor;

// ARC 캐시
typedef struct {
    LRUList T1;
    LRUList T2;
    LRUList B1;
    LRUList B2;
    int p;
    int cache_size;
    long global_tick;
    int recent_evictions;
    int access_window;
    int ghost_hit_history[GHOST_HIT_WINDOW];
    int ghost_hit_index;
    ZoneInfo zones[1024];
    int total_zones;
    int active_zones_count;
    long total_zone_resets;
    long total_pages_migrated;
    long hits;
    long misses;
    PageEntryPool* page_pool;
    DeferredEvictionQueue* deferred_queue;
} ARCCache;


extern ARCCache* global_arc_cache;
// ============================================================
// 함수 프로토타입
// ============================================================

// 해시테이블
HashTable* ht_create(void);
void ht_destroy(HashTable* table);
PageEntry* ht_get(HashTable* table, int page_id);
bool ht_set(HashTable* table, int page_id, PageEntry* entry);
void ht_remove(HashTable* table, int page_id);

// 메모리 풀
PageEntryPool* pool_create(int capacity);
void pool_destroy(PageEntryPool* pool);
PageEntry* pool_alloc(PageEntryPool* pool);
void pool_free(PageEntryPool* pool, PageEntry* entry);

// Zone Lock
bool zone_lock_acquire(ARCCache* cache, int zone_id);
void zone_lock_release(ARCCache* cache, int zone_id);

// Deferred Eviction Queue
DeferredEvictionQueue* deferred_queue_create(void);
void deferred_queue_destroy(DeferredEvictionQueue* queue);
void enqueue_deferred_eviction(ARCCache* cache, PageEntry** batch,
                                int count, int zone, int source_type);
void process_deferred_queue(ARCCache* cache);

// Zone 관리
int get_zone_id(int page_id);
int count_zone_pages(ARCCache* cache, int zone_id);
void update_zone_info(ARCCache* cache, int zone_id, int delta);
int find_alternative_eviction_zone(ARCCache* cache, int locked_zone);
void rebuild_batch_for_zone(ARCCache* cache, LRUList* list, int target_zone,
                             PageEntry** batch, int* batch_count);

// 캐시 함수
ARCCache* init_arc_cache(int size);
void free_arc_cache(ARCCache* cache);
void arc_access(ARCCache* cache, int page_id);
void replace(ARCCache* cache, int in_b2);
void replace_zone_aware(ARCCache* cache, int in_b2);
void print_zns_statistics(ARCCache* cache);

// ML Predictor
float predict_reuse_probability_safe(PageFeatures* features);

// 성능 모니터
extern PerformanceMonitor perf_mon;

#endif // ZNS_ARC_FINAL_SEQLOCK_H
