#include "zns_arc_final_seqlock.h"
#include "arc_ml_predictor_seqlock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <time.h>
#include <sched.h>
#include <limits.h>


//=================================================================
// 글로벌 ARC 캐시 인스턴스
//=================================================================
ARCCache* global_arc_cache = NULL;

//=================================================================
// 자동 초기화 (QEMU 시작 시 실행)
//=================================================================
static void __attribute__((constructor)) femu_arc_auto_init(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "================================================\n");
    fprintf(stderr, "[FEMU-ARC] Auto-initialization starting...\n");
    fprintf(stderr, "================================================\n");
    fflush(stderr);
    
    if (!global_arc_cache) {
        global_arc_cache = init_arc_cache(262144);  // 262144 pages = 1GB
        //global_arc_cache = init_arc_cache(1000); //testing Pages Migrated
        if (global_arc_cache) {
            fprintf(stderr, "[FEMU-ARC] Cache initialized successfully\n");
            fprintf(stderr, "[FEMU-ARC] Cache size: 262144 pages (1 GB)\n");
        } else {
            fprintf(stderr, "[FEMU-ARC] ERROR: Cache initialization failed!\n");
        }
    }
    
    fprintf(stderr, "================================================\n\n");
    fflush(stderr);
}

static void __attribute__((destructor)) femu_arc_auto_cleanup(void) {
    fprintf(stderr, "[FEMU-ARC] Auto-cleanup starting...\n");
    if (global_arc_cache) {
        // 통계 먼저 출력
        print_zns_statistics(global_arc_cache);
        
        // 메모리 해제
        free_arc_cache(global_arc_cache);
        global_arc_cache = NULL;
    }
    fprintf(stderr, "[FEMU-ARC] Auto-cleanup completed\n");
}
//=================================================================
// 자동 해제 (QEMU 종료 시 실행)
//=================================================================
static void __attribute__((destructor)) femu_arc_cleanup(void) {
    if (global_arc_cache) {
        fprintf(stderr, "[FEMU-ARC] Cleanup started\n");
        print_zns_statistics(global_arc_cache); // 종료 시 최종 통계 출력
        free_arc_cache(global_arc_cache);
        global_arc_cache = NULL;
        fprintf(stderr, "[FEMU-ARC] Cleanup complete\n");
    }
}


// ============================================================
// 전역 성능 모니터
// ============================================================
PerformanceMonitor perf_mon = {0};

// ============================================================
// 유틸리티 함수
// ============================================================
static int arc_max(int a, int b) { return (a > b) ? a : b; }
static int arc_min(int a, int b) { return (a < b) ? a : b; }

static inline long get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

// ============================================================
// 해시테이블 구현
// ============================================================
static uint64_t hash_key(int page_id) {
    uint64_t hash = FNV_OFFSET;
    uint8_t* bytes = (uint8_t*)&page_id;
    for (size_t i = 0; i < sizeof(int); i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

HashTable* ht_create(void) {
    HashTable* table = (HashTable*)malloc(sizeof(HashTable));
    if (!table) return NULL;
    table->capacity = HASH_INITIAL_CAPACITY;
    table->length = 0;
    table->entries = (HashEntry*)calloc(table->capacity, sizeof(HashEntry));
    for (size_t i = 0; i < table->capacity; i++) 
        table->entries[i].page_id = -1;
    return table;
}

void ht_destroy(HashTable* table) {
    if (!table) return;
    free(table->entries);
    free(table);
}

static bool ht_expand(HashTable* table) {
    size_t new_capacity = table->capacity * 2;
    HashEntry* new_entries = (HashEntry*)calloc(new_capacity, sizeof(HashEntry));
    if (!new_entries) return false;

    for (size_t i = 0; i < new_capacity; i++) 
        new_entries[i].page_id = -1;

    for (size_t i = 0; i < table->capacity; i++) {
        if (table->entries[i].page_id != -1) {
            uint64_t hash = hash_key(table->entries[i].page_id);
            size_t index = (size_t)(hash & (uint64_t)(new_capacity - 1));
            while (new_entries[index].page_id != -1) {
                index = (index + 1) & (new_capacity - 1);
            }
            new_entries[index] = table->entries[i];
        }
    }

    free(table->entries);
    table->entries = new_entries;
    table->capacity = new_capacity;
    return true;
}

PageEntry* ht_get(HashTable* table, int page_id) {
    if (!table) return NULL;
    uint64_t hash = hash_key(page_id);
    size_t index = (size_t)(hash & (uint64_t)(table->capacity - 1));

    while (table->entries[index].page_id != -1) {
        if (table->entries[index].page_id == page_id) 
            return table->entries[index].entry;
        index = (index + 1) & (table->capacity - 1);
    }
    return NULL;
}

bool ht_set(HashTable* table, int page_id, PageEntry* entry) {
    if (table->length >= table->capacity / 2) ht_expand(table);

    uint64_t hash = hash_key(page_id);
    size_t index = (size_t)(hash & (uint64_t)(table->capacity - 1));

    while (table->entries[index].page_id != -1) {
        if (table->entries[index].page_id == page_id) {
            table->entries[index].entry = entry;
            return true;
        }
        index = (index + 1) & (table->capacity - 1);
    }

    table->entries[index].page_id = page_id;
    table->entries[index].entry = entry;
    table->length++;
    return true;
}

void ht_remove(HashTable* table, int page_id) {
    uint64_t hash = hash_key(page_id);
    size_t index = (size_t)(hash & (uint64_t)(table->capacity - 1));

    while (table->entries[index].page_id != -1) {
        if (table->entries[index].page_id == page_id) {
            table->entries[index].page_id = -1;
            table->entries[index].entry = NULL;
            table->length--;
            return;
        }
        index = (index + 1) & (table->capacity - 1);
    }
}

// ============================================================
// 메모리 풀 구현
// ============================================================
PageEntryPool* pool_create(int capacity) {
    PageEntryPool* pool = (PageEntryPool*)malloc(sizeof(PageEntryPool));
    if (!pool) return NULL;

    pool->pool = (PageEntry*)calloc(capacity, sizeof(PageEntry));
    if (!pool->pool) {
        free(pool);
        return NULL;
    }

    int bitmap_size = (capacity + 31) / 32;
    pool->free_bitmap = (atomic_uint*)malloc(bitmap_size * sizeof(atomic_uint));
    for (int i = 0; i < bitmap_size; i++) {
        atomic_init(&pool->free_bitmap[i], 0xFFFFFFFF);
    }

    int remainder = capacity % 32;
    if (remainder != 0) {
        atomic_init(&pool->free_bitmap[bitmap_size - 1], 
                   (1U << remainder) - 1);
    }

    atomic_init(&pool->pool_size, 0);
    pool->capacity = capacity;

    return pool;
}

void pool_destroy(PageEntryPool* pool) {
    if (!pool) return;

    for (int i = 0; i < pool->capacity; i++) {
        if (pool->pool[i].feat.reuse_history) {
            free(pool->pool[i].feat.reuse_history);
        }
    }

    free(pool->pool);
    free(pool->free_bitmap);
    free(pool);
}

PageEntry* pool_alloc(PageEntryPool* pool) {
    if (!pool) return NULL;

    int current_size = atomic_load(&pool->pool_size);
    float utilization = (float)current_size / pool->capacity;

    if (utilization > POOL_FALLBACK_THRESHOLD) {
        fprintf(stderr, "[WARN] Memory pool utilization: %.1f%%\n", 
                utilization * 100);
    }

    int bitmap_size = (pool->capacity + 31) / 32;

    for (int i = 0; i < bitmap_size; i++) {
        unsigned int bitmap = atomic_load(&pool->free_bitmap[i]);

        while (bitmap != 0) {
            int bit = __builtin_ffs(bitmap) - 1;
            unsigned int mask = ~(1U << bit);

            if (atomic_compare_exchange_weak(&pool->free_bitmap[i], 
                                            &bitmap, 
                                            bitmap & mask)) {
                int idx = i * 32 + bit;
                if (idx >= pool->capacity) return NULL;

                PageEntry* entry = &pool->pool[idx];
                memset(entry, 0, sizeof(PageEntry));
    seqlock_init(&entry->feat.lock);
                entry->in_pool = true;

                atomic_fetch_add(&pool->pool_size, 1);
                return entry;
            }
        }
    }

    atomic_fetch_add(&perf_mon.pool_fallback_count, 1);
    fprintf(stderr, "[CRITICAL] Pool exhausted! Using malloc fallback (count: %ld)\n",
            atomic_load(&perf_mon.pool_fallback_count));

    PageEntry* entry = (PageEntry*)calloc(1, sizeof(PageEntry));
    if (entry) {
        entry->in_pool = false;
    }
    return entry;
}

void pool_free(PageEntryPool* pool, PageEntry* entry) {
    if (!pool || !entry) return;

    if (!entry->in_pool) {
        if (entry->feat.reuse_history) free(entry->feat.reuse_history);
        free(entry);
        return;
    }

    if (entry->feat.reuse_history) {
        free(entry->feat.reuse_history);
        entry->feat.reuse_history = NULL;
    }

    int idx = entry - pool->pool;
    if (idx < 0 || idx >= pool->capacity) return;

    int bitmap_idx = idx / 32;
    int bit = idx % 32;
    atomic_fetch_or(&pool->free_bitmap[bitmap_idx], 1U << bit);

    atomic_fetch_sub(&pool->pool_size, 1);
}

// ============================================================
// Zone Lock (Timeout 포함)
// ============================================================
bool zone_lock_acquire(ARCCache* cache, int zone_id) {
    if (zone_id < 0 || zone_id >= 1024) return false;

    long start_time = get_time_ns();
    long spin_count = 0;

    while (atomic_flag_test_and_set(&cache->zones[zone_id].zone_lock)) {
        spin_count++;

        long elapsed = get_time_ns() - start_time;
        if (elapsed > SPINLOCK_TIMEOUT_NS) {
            atomic_fetch_add(&perf_mon.timeout_count, 1);
            return false;
        }

        if (spin_count % 100 == 0) {
            sched_yield();
        }
    }

    long total_time = get_time_ns() - start_time;
    atomic_fetch_add(&perf_mon.total_spins, spin_count);
    atomic_fetch_add(&perf_mon.zone_contention[zone_id], 1);

    long current_max = atomic_load(&perf_mon.max_spin_ns);
    while (total_time > current_max) {
        if (atomic_compare_exchange_weak(&perf_mon.max_spin_ns, 
                                         &current_max, total_time))
            break;
    }

    return true;
}

void zone_lock_release(ARCCache* cache, int zone_id) {
    if (zone_id < 0 || zone_id >= 1024) return;
    atomic_flag_clear(&cache->zones[zone_id].zone_lock);
}

// ============================================================
// Deferred Eviction Queue
// ============================================================
DeferredEvictionQueue* deferred_queue_create(void) {
    DeferredEvictionQueue* queue = 
        (DeferredEvictionQueue*)calloc(1, sizeof(DeferredEvictionQueue));
    if (!queue) return NULL;

    atomic_init(&queue->head, 0);
    atomic_init(&queue->tail, 0);
    atomic_init(&queue->size, 0);

    return queue;
}

void deferred_queue_destroy(DeferredEvictionQueue* queue) {
    if (queue) free(queue);
}

void enqueue_deferred_eviction(ARCCache* cache, PageEntry** batch, 
                               int count, int zone, int source_type) {
    int current_size = atomic_load(&cache->deferred_queue->size);

    if (current_size >= DEFERRED_QUEUE_SIZE) {
        fprintf(stderr, "[WARN] Deferred queue full, processing immediately\n");
        process_deferred_queue(cache);
    }

    int tail = atomic_load(&cache->deferred_queue->tail);
    DeferredBatch* db = &cache->deferred_queue->queue[tail];

    memcpy(db->pages, batch, count * sizeof(PageEntry*));
    db->count = count;
    db->target_zone = zone;
    db->source_type = source_type;
    db->timestamp = cache->global_tick;

    atomic_store(&cache->deferred_queue->tail, (tail + 1) % DEFERRED_QUEUE_SIZE);
    atomic_fetch_add(&cache->deferred_queue->size, 1);
    atomic_fetch_add(&perf_mon.deferred_eviction_count, 1);
}

void process_deferred_queue(ARCCache* cache) {
    int processed = 0;
    int current_size = atomic_load(&cache->deferred_queue->size);

    while (current_size > 0 && processed < 8) {
        int head = atomic_load(&cache->deferred_queue->head);
        DeferredBatch* db = &cache->deferred_queue->queue[head];

        if (zone_lock_acquire(cache, db->target_zone)) {
            for(int i = 0; i < db->count; i++) {
                PageEntry* e = db->pages[i];
                if (e && e->zone_id == db->target_zone) {
                    update_zone_info(cache, e->zone_id, -1);

                    LRUList* source = (db->source_type == 1) ? &cache->T1 : &cache->T2;
                    LRUList* ghost = (db->source_type == 1) ? &cache->B1 : &cache->B2;

                    PageEntry* found = ht_get(source->hash, e->page_id);
                    if (found == e) {
                        if (e->prev) e->prev->next = e->next;
                        else source->head = e->next;
                        if (e->next) e->next->prev = e->prev;
                        else source->tail = e->prev;
                        ht_remove(source->hash, e->page_id);
                        source->size--;

                        e->next = ghost->head;
                        e->prev = NULL;
                        if (ghost->head) ghost->head->prev = e;
                        ghost->head = e;
                        if (!ghost->tail) ghost->tail = e;
                        ht_set(ghost->hash, e->page_id, e);
                        ghost->size++;
                    }

                    cache->total_pages_migrated++;
                }
            }

            zone_lock_release(cache, db->target_zone);

            atomic_store(&cache->deferred_queue->head, 
                        (head + 1) % DEFERRED_QUEUE_SIZE);
            atomic_fetch_sub(&cache->deferred_queue->size, 1);
            atomic_fetch_add(&perf_mon.deferred_processed_count, 1);
            processed++;
        } else {
            break;
        }

        current_size = atomic_load(&cache->deferred_queue->size);
    }
}

// ============================================================
// Zone 관리
// ============================================================
int get_zone_id(int page_id) { 
    return page_id / PAGES_PER_ZONE; 
}

int count_zone_pages(ARCCache* cache, int zone_id) {
    if (zone_id < 0 || zone_id >= 1024) return 0;
    return cache->zones[zone_id].valid_page_count;
}

void update_zone_info(ARCCache* cache, int zone_id, int delta) {
    if (zone_id < 0 || zone_id >= 1024) return;
    if (zone_id >= cache->total_zones) cache->total_zones = zone_id + 1;

    cache->zones[zone_id].zone_id = zone_id;
    cache->zones[zone_id].valid_page_count += delta;

    if (delta > 0 && cache->zones[zone_id].valid_page_count == 1) {
        cache->zones[zone_id].is_active = true;
        cache->active_zones_count++;
    }

    if (cache->zones[zone_id].valid_page_count <= 0) {
        cache->zones[zone_id].valid_page_count = 0;
        if (cache->zones[zone_id].is_active) {
            cache->zones[zone_id].is_active = false;
            cache->active_zones_count--;
        }
    }
}

int find_alternative_eviction_zone(ARCCache* cache, int locked_zone) {
    int min_pages = INT_MAX;
    int best_zone = -1;

    for (int i = 0; i < cache->total_zones; i++) {
        if (i == locked_zone) continue;
        if (!cache->zones[i].is_active) continue;

        if (atomic_flag_test_and_set(&cache->zones[i].zone_lock)) {
            atomic_flag_clear(&cache->zones[i].zone_lock);
            continue;
        }
        atomic_flag_clear(&cache->zones[i].zone_lock);

        int pages = cache->zones[i].valid_page_count;
        if (pages < min_pages && pages < PAGES_PER_ZONE / 4) {
            min_pages = pages;
            best_zone = i;
        }
    }

    return best_zone;
}

void rebuild_batch_for_zone(ARCCache* cache, LRUList* list, int target_zone,
                            PageEntry** batch, int* batch_count) {
    *batch_count = 0;
    PageEntry* curr = list->tail;
    int safety = 0;

    while(curr && *batch_count < ZONE_BATCH_SIZE) {
        if (++safety > 5000) break;

        if (curr->zone_id == target_zone) {
            batch[(*batch_count)++] = curr;
        }
        curr = curr->prev;
    }
}

// ============================================================
// ML 피처 추출 및 통계
// ============================================================
static int calculate_rank(LRUList* list, PageEntry* target) {
    int rank = 0;
    PageEntry* p = list->head;
    int limit = 0;

    while (p && p != target) {
        rank++;
        p = p->next;
        if (++limit > 20000) break;
    }
    return rank;
}

static void count_ghost_hits(ARCCache* cache, int* b1, int* b2) {
    *b1 = 0; *b2 = 0;
    for(int i = 0; i < GHOST_HIT_WINDOW; i++) {
        if(cache->ghost_hit_history[i] == 1) (*b1)++;
        else if(cache->ghost_hit_history[i] == 2) (*b2)++;
    }
}

static void update_reuse_stats(ARCCache* cache, PageEntry* entry) {
    seqlock_write_begin(&entry->feat.lock);  // 쓰기 시작

    long interval = cache->global_tick - entry->feat.last_access_tick;
    entry->feat.last_reuse_distance = interval;
    entry->feat.reuse_count++;

    if (entry->feat.avg_reuse_interval == 0)
        entry->feat.avg_reuse_interval = (float)interval;
    else
        entry->feat.avg_reuse_interval =
            0.7f * entry->feat.avg_reuse_interval + 0.3f * interval;

    float diff = interval - entry->feat.avg_reuse_interval;
    entry->feat.reuse_variance =
        0.7f * entry->feat.reuse_variance + 0.3f * (diff * diff);

    if (!entry->feat.reuse_history) {
        entry->feat.reuse_history = (long*)calloc(5, sizeof(long));
        entry->feat.history_size = 0;
    }

    if (entry->feat.history_size < 5)
        entry->feat.reuse_history[entry->feat.history_size++] = interval;
    else {
        for(int i = 0; i < 4; i++)
            entry->feat.reuse_history[i] = entry->feat.reuse_history[i+1];
        entry->feat.reuse_history[4] = interval;
    }

    seqlock_write_end(&entry->feat.lock);  // 쓰기 종료


    if (entry->feat.history_size < 5)
        entry->feat.reuse_history[entry->feat.history_size++] = interval;
    else {
        for(int i = 0; i < 4; i++)
            entry->feat.reuse_history[i] = entry->feat.reuse_history[i+1];
        entry->feat.reuse_history[4] = interval;
    }
}

static void calculate_neighbors(ARCCache* cache, PageEntry* entry) {
    float sum_acc = 0, sum_idle = 0;
    int count = 0;

    // 이웃 노드 읽기 (seqlock으로 보호)
    PageEntry* n = entry->prev;
    for(int i = 0; i < 3 && n; i++, n = n->prev) {
        uint32_t seq;
        do {
            seq = seqlock_read_begin(&n->feat.lock);
            sum_acc += n->feat.access_count;
            sum_idle += (cache->global_tick - n->feat.last_access_tick);
        } while(seqlock_read_retry(&n->feat.lock, seq));
        count++;
    }

    n = entry->next;
    for(int i = 0; i < 3 && n; i++, n = n->next) {
        uint32_t seq;
        do {
            seq = seqlock_read_begin(&n->feat.lock);
            sum_acc += n->feat.access_count;
            sum_idle += (cache->global_tick - n->feat.last_access_tick);
        } while(seqlock_read_retry(&n->feat.lock, seq));
        count++;
    }

    // 결과 저장 (seqlock으로 보호)
    seqlock_write_begin(&entry->feat.lock);
    entry->feat.neighbor_avg_access = (count > 0) ? sum_acc / count : 0;
    entry->feat.neighbor_avg_idle = (count > 0) ? sum_idle / count : 0;
    seqlock_write_end(&entry->feat.lock);


    n = entry->next;
    for(int i = 0; i < 3 && n; i++, n = n->next) {
        sum_acc += n->feat.access_count;
        sum_idle += (cache->global_tick - n->feat.last_access_tick);
        count++;
    }

    entry->feat.neighbor_avg_access = (count > 0) ? sum_acc / count : 0;
    entry->feat.neighbor_avg_idle = (count > 0) ? sum_idle / count : 0;
}

static void extract_features_for_ml(ARCCache* cache, LRUList* list, 
                             PageEntry* entry, float* features) {
    int i = 0;
    int rank = calculate_rank(list, entry);
    float pos_ratio = (list->size > 1) ? (float)rank / (list->size - 1) : 0.0f;
    int b1_hits, b2_hits;
    count_ghost_hits(cache, &b1_hits, &b2_hits);
    calculate_neighbors(cache, entry);
    int zone_pages = count_zone_pages(cache, entry->zone_id);

    float zone_valid_ratio = 0.0f;
    if (entry->zone_id >= 0 && entry->zone_id < 1024) {
        if (cache->zones[entry->zone_id].total_page_count > 0) {
            zone_valid_ratio = (float)cache->zones[entry->zone_id].valid_page_count /
                              cache->zones[entry->zone_id].total_page_count;
        }
    }

    features[i++] = (float)entry->feat.access_count;
    features[i++] = (float)(cache->global_tick - entry->feat.last_access_tick);
    features[i++] = (float)entry->feat.list_type;
    features[i++] = pos_ratio;
    features[i++] = (float)entry->feat.last_reuse_distance;
    features[i++] = entry->feat.avg_reuse_interval;
    features[i++] = entry->feat.reuse_variance;
    features[i++] = (float)entry->feat.reuse_count;
    features[i++] = (float)cache->p;
    features[i++] = (float)cache->B1.size;
    features[i++] = (float)cache->B2.size;
    features[i++] = (float)b1_hits;
    features[i++] = (float)b2_hits;
    features[i++] = (float)(cache->T1.size + cache->T2.size) / cache->cache_size;
    features[i++] = (float)rank;
    features[i++] = (cache->access_window > 0) ? 
                    (float)cache->recent_evictions / cache->access_window : 0.0f;
    features[i++] = entry->feat.neighbor_avg_access;
    features[i++] = entry->feat.neighbor_avg_idle;
    features[i++] = (float)entry->zone_id;
    features[i++] = zone_valid_ratio;
    features[i++] = (float)zone_pages;
    features[i++] = (float)cache->active_zones_count;
}

// ============================================================
// 리스트 조작 함수
// ============================================================
static void init_list(LRUList* list) {
    list->head = list->tail = NULL;
    list->size = 0;
    list->hash = ht_create();
}

static PageEntry* find_page(LRUList* list, int page_id) {
    return ht_get(list->hash, page_id);
}

static void remove_from_list(LRUList* list, PageEntry* entry) {
    if(entry->prev) entry->prev->next = entry->next;
    else list->head = entry->next;

    if(entry->next) entry->next->prev = entry->prev;
    else list->tail = entry->prev;

    ht_remove(list->hash, entry->page_id);
    list->size--;
}

static void add_to_mru(LRUList* list, PageEntry* entry) {
    entry->next = list->head;
    entry->prev = NULL;

    if(list->head) list->head->prev = entry;
    list->head = entry;

    if(!list->tail) list->tail = entry;

    ht_set(list->hash, entry->page_id, entry);
    list->size++;
}

static PageEntry* remove_lru(LRUList* list) {
    if(!list->tail) return NULL;

    PageEntry* lru = list->tail;

    if(lru->prev) lru->prev->next = NULL;
    else list->head = NULL;

    list->tail = lru->prev;
    ht_remove(list->hash, lru->page_id);
    list->size--;

    return lru;
}

static void free_list(LRUList* list) {
    PageEntry* e = list->head;
    while(e) {
        PageEntry* n = e->next;
        if(e->feat.reuse_history) free(e->feat.reuse_history);
        if (!e->in_pool) free(e);
        e = n;
    }
    ht_destroy(list->hash);
}

// ============================================================
// Zone-Aware 교체 정책 (3-Tier Fallback)
// ============================================================
void replace_zone_aware(ARCCache* cache, int in_b2) {
    PageEntry* victim;
    LRUList* source_list;
    int source_type;
    int rescue_limit = 3;

    if (cache->T1.size > 0 && 
        ((in_b2 && cache->T1.size == cache->p) || (cache->T1.size > cache->p))) {
        source_list = &cache->T1;
        source_type = 1;
    } else {
        source_list = &cache->T2;
        source_type = 2;
    }

    // 1. Rescue Loop
    while (rescue_limit > 0) {
        victim = source_list->tail;
        if (!victim) return;

        float features[22];
        extract_features_for_ml(cache, source_list, victim, features);
        float reuse_prob = predict_reuse_probability_safe(&victim->feat);

        if (reuse_prob > 0.6f) {
            remove_from_list(source_list, victim);
            add_to_mru(source_list, victim);
            rescue_limit--;
            continue;
        }
        break;
    }

    // 2. Batch Collection
    int victim_zone = victim->zone_id;
    PageEntry* batch[ZONE_BATCH_SIZE];
    int batch_count = 0;
    PageEntry* curr = victim;

    int safety = 0;
    while(curr && batch_count < ZONE_BATCH_SIZE) {
        if (++safety > 1000) break;

        int cz = curr->zone_id;
        bool is_valid_low = false;

        if (cz >= 0 && cz < 1024) {
            if (cache->zones[cz].valid_page_count < 10) 
                is_valid_low = true;
        }

        if (cz == victim_zone || is_valid_low) {
            batch[batch_count++] = curr;
        }

        curr = curr->prev;
        if (curr == victim) break;
    }

    // 3. [Tier 1] Try-Lock (Fast Path)
    if (victim_zone >= 0 && victim_zone < 1024) {
        if (!atomic_flag_test_and_set(&cache->zones[victim_zone].zone_lock)) {
            // Fast path: Lock acquired immediately
            for(int i = 0; i < batch_count; i++) {
                PageEntry* e = batch[i];
                update_zone_info(cache, e->zone_id, -1);
                remove_from_list(source_list, e);

                if(source_type == 1) add_to_mru(&cache->B1, e);
                else add_to_mru(&cache->B2, e);

                cache->total_pages_migrated++;
            }
            atomic_flag_clear(&cache->zones[victim_zone].zone_lock);
            goto cleanup;
        }

        // 4. [Tier 2] Zone Rotation
        int alt_zone = find_alternative_eviction_zone(cache, victim_zone);
        if (alt_zone >= 0) {
            atomic_fetch_add(&perf_mon.zone_rotation_count, 1);

            rebuild_batch_for_zone(cache, source_list, alt_zone, batch, &batch_count);

            if (batch_count > 0 && zone_lock_acquire(cache, alt_zone)) {
                for(int i = 0; i < batch_count; i++) {
                    PageEntry* e = batch[i];
                    update_zone_info(cache, e->zone_id, -1);
                    remove_from_list(source_list, e);

                    if(source_type == 1) add_to_mru(&cache->B1, e);
                    else add_to_mru(&cache->B2, e);

                    cache->total_pages_migrated++;
                }
                zone_lock_release(cache, alt_zone);
                goto cleanup;
            }
        }

        // 5. [Tier 3] Deferred Eviction
        enqueue_deferred_eviction(cache, batch, batch_count, victim_zone, source_type);
    }

cleanup:
    // Ghost list 정리
    safety = 0;
    while (cache->B1.size + cache->B2.size > 2 * cache->cache_size) {
        if (++safety > 20000) break;

        if (cache->B1.size > cache->p) {
            PageEntry* old = remove_lru(&cache->B1);
            if (!old) break;
            pool_free(cache->page_pool, old);
        } else if (cache->B2.size > 0) {
            PageEntry* old = remove_lru(&cache->B2);
            if (!old) break;
            pool_free(cache->page_pool, old);
        } else break;
    }
}

void replace(ARCCache* cache, int in_b2) {
    if ((float)(cache->T1.size + cache->T2.size) / cache->cache_size > 0.85) {
        replace_zone_aware(cache, in_b2);
        return;
    }

    PageEntry* victim;
    if (cache->T1.size > 0 && 
        ((in_b2 && cache->T1.size == cache->p) || (cache->T1.size > cache->p))) {
        victim = remove_lru(&cache->T1);
        if(victim) {
            update_zone_info(cache, victim->zone_id, -1);
            add_to_mru(&cache->B1, victim);
        }
    } else {
        victim = remove_lru(&cache->T2);
        if(victim) {
            update_zone_info(cache, victim->zone_id, -1);
            add_to_mru(&cache->B2, victim);
        }
    }
}

static PageEntry* create_page(ARCCache* cache, int page_id) {
    PageEntry* e = pool_alloc(cache->page_pool);
    if (!e) return NULL;

    e->page_id = page_id;
    e->zone_id = get_zone_id(page_id);
    e->feat.access_count = 1;
    e->feat.last_access_tick = cache->global_tick;
    e->feat.list_type = 1;

    int zid = e->zone_id;
    if (zid >= 0 && zid < 1024) {
        if (zone_lock_acquire(cache, zid)) {
            update_zone_info(cache, zid, 1);
            cache->zones[zid].total_page_count++;
            zone_lock_release(cache, zid);
        } else {
            update_zone_info(cache, zid, 1);
            cache->zones[zid].total_page_count++;
        }
    }

    return e;
}

// ============================================================
// ARC Access
// ============================================================
void arc_access(ARCCache* cache, int page_id) {
    // Deferred queue 주기적 처리
    // [동작 확인용] 100,000회 접근마다 통계 출력 (I/O 부하에 따라 조절 가능)
    /*
    if ((cache->hits + cache->misses) > 0 && (cache->hits + cache->misses) % 100000 == 0) {
        print_zns_statistics(cache);
        fflush(stderr); // 버퍼 강제 비우기 (로그 즉시 확인용)
    }
    */

    if (cache->access_window % 1000 == 0 && cache->access_window > 0) {
        process_deferred_queue(cache);
    }

    PageEntry* _chk = find_page(&cache->T1, page_id);
    if (!_chk) _chk = find_page(&cache->T2, page_id);
    if (_chk) cache->hits++; 
    else cache->misses++;

    cache->global_tick++;
    cache->access_window++;

    if(cache->access_window > 1000) {
        cache->recent_evictions = 0;
        cache->access_window = 0;
    }

    PageEntry* e = find_page(&cache->T1, page_id);
    if(e) {
        update_reuse_stats(cache, e);
        remove_from_list(&cache->T1, e);
        add_to_mru(&cache->T2, e);
        e->feat.list_type = 2;
        seqlock_write_begin(&e->feat.lock);
        e->feat.access_count++;
        e->feat.last_access_tick = cache->global_tick;
        seqlock_write_end(&e->feat.lock);
        return;
    }

    e = find_page(&cache->T2, page_id);
    if(e) {
        update_reuse_stats(cache, e);
        remove_from_list(&cache->T2, e);
        add_to_mru(&cache->T2, e);
        seqlock_write_begin(&e->feat.lock);
        e->feat.access_count++;
        e->feat.last_access_tick = cache->global_tick;
        seqlock_write_end(&e->feat.lock);
        return;
    }

    e = find_page(&cache->B1, page_id);
    if(e) {
        cache->p = arc_min(cache->p + 
                          (cache->B2.size >= cache->B1.size ? 
                           1 : cache->B2.size / cache->B1.size), 
                          cache->cache_size);
        replace(cache, 0);
        remove_from_list(&cache->B1, e);
        e->feat.access_count = 1;
        e->feat.last_access_tick = cache->global_tick;
        e->feat.list_type = 2;
        update_zone_info(cache, e->zone_id, 1);
        add_to_mru(&cache->T2, e);
        return;
    }

    e = find_page(&cache->B2, page_id);
    if(e) {
        cache->p = arc_max(cache->p - 
                          (cache->B1.size >= cache->B2.size ? 
                           1 : cache->B1.size / cache->B2.size), 
                          0);
        replace(cache, 1);
        remove_from_list(&cache->B2, e);
        e->feat.access_count = 1;
        e->feat.last_access_tick = cache->global_tick;
        e->feat.list_type = 2;
        update_zone_info(cache, e->zone_id, 1);
        add_to_mru(&cache->T2, e);
        return;
    }

    if(cache->T1.size + cache->B1.size == cache->cache_size) {
        if(cache->T1.size < cache->cache_size) {
            PageEntry* o = remove_lru(&cache->B1);
            if(o) pool_free(cache->page_pool, o);
            replace(cache, 0);
        } else {
            PageEntry* o = remove_lru(&cache->T1);
            if(o) {
                update_zone_info(cache, o->zone_id, -1);
                pool_free(cache->page_pool, o);
                cache->recent_evictions++;
            }
        }
    } else {
        if(cache->T1.size + cache->T2.size + cache->B1.size + cache->B2.size 
           >= cache->cache_size) {
            if(cache->T1.size + cache->T2.size + cache->B1.size + cache->B2.size 
               == 2 * cache->cache_size) {
                PageEntry* o = remove_lru(&cache->B2);
                if(o) pool_free(cache->page_pool, o);
            }
            replace(cache, 0);
        }
    }

    e = create_page(cache, page_id);
    if (e) add_to_mru(&cache->T1, e);
}

// ============================================================
// 초기화 및 정리
// ============================================================
ARCCache* init_arc_cache(int size) {
    ARCCache* c = (ARCCache*)calloc(1, sizeof(ARCCache));

    init_list(&c->T1);
    init_list(&c->T2);
    init_list(&c->B1);
    init_list(&c->B2);

    c->cache_size = size;
    c->p = size / 2;

    c->page_pool = pool_create(POOL_SIZE);
    if (!c->page_pool) {
        fprintf(stderr, "Failed to create memory pool!\n");
        free(c);
        return NULL;
    }

    c->deferred_queue = deferred_queue_create();
    if (!c->deferred_queue) {
        fprintf(stderr, "Failed to create deferred queue!\n");
        pool_destroy(c->page_pool);
        free(c);
        return NULL;
    }

    for (int i = 0; i < 1024; i++) {
        atomic_flag_clear(&c->zones[i].zone_lock);
    }

    atomic_init(&perf_mon.total_spins, 0);
    atomic_init(&perf_mon.timeout_count, 0);
    atomic_init(&perf_mon.max_spin_ns, 0);
    atomic_init(&perf_mon.pool_fallback_count, 0);
    atomic_init(&perf_mon.zone_rotation_count, 0);
    atomic_init(&perf_mon.deferred_eviction_count, 0);
    atomic_init(&perf_mon.deferred_processed_count, 0);
    for (int i = 0; i < 1024; i++) {
        atomic_init(&perf_mon.zone_contention[i], 0);
    }

    init_ml_predictor(NULL);

    fprintf(stderr, "\n=== ZNS-ARC Initialization ===\n");
    fprintf(stderr, "Cache Size: %d pages\n", size);
    fprintf(stderr, "Memory Pool: %d entries (%.2f MB)\n", POOL_SIZE, 
           POOL_SIZE * sizeof(PageEntry) / 1024.0 / 1024.0);
    fprintf(stderr, "Deferred Queue: %d slots\n", DEFERRED_QUEUE_SIZE);
    fprintf(stderr, "Zone Batch Size: %d\n", ZONE_BATCH_SIZE);
    fprintf(stderr, "Spinlock Timeout: %d μs\n", SPINLOCK_TIMEOUT_NS / 1000);
    fprintf(stderr, "Features: Lock-Free Pool + Zone Rotation + Deferred Eviction\n\n");

    c->hits = 0;
    c->misses = 0;

    return c;
}

void free_arc_cache(ARCCache* c) {
    free_list(&c->T1);
    free_list(&c->T2);
    free_list(&c->B1);
    free_list(&c->B2);
    pool_destroy(c->page_pool);
    deferred_queue_destroy(c->deferred_queue);
    free_ml_predictor();
    free(c);
}

void print_zns_statistics(ARCCache* c) {
    fprintf(stderr, "\n=== ZNS-ARC Statistics ===\n");

    fprintf(stderr, "\n[Zone Stats]\n");
    fprintf(stderr, "  Pages Migrated: %ld\n", c->total_pages_migrated);
    fprintf(stderr, "  Active Zones: %d / %d\n", c->active_zones_count, c->total_zones);

    fprintf(stderr, "\n[Cache Performance]\n");
    fprintf(stderr, "  Hits: %ld\n", c->hits);
    fprintf(stderr, "  Misses: %ld\n", c->misses);
    fprintf(stderr, "  Hit Rate: %.2f%%\n", 
           (c->hits + c->misses > 0) ? 
           100.0 * c->hits / (c->hits + c->misses) : 0.0);

    fprintf(stderr, "\n[Memory Pool]\n");
    int pool_size = atomic_load(&c->page_pool->pool_size);
    fprintf(stderr, "  Usage: %d / %d (%.1f%%)\n", 
           pool_size, c->page_pool->capacity,
           100.0 * pool_size / c->page_pool->capacity);
    fprintf(stderr, "  Fallback Count: %ld\n", 
           atomic_load(&perf_mon.pool_fallback_count));

    fprintf(stderr, "\n[Lock Performance]\n");
    fprintf(stderr, "  Total Spins: %ld\n", atomic_load(&perf_mon.total_spins));
    fprintf(stderr, "  Timeout Events: %ld\n", atomic_load(&perf_mon.timeout_count));
    fprintf(stderr, "  Max Spin Time: %.2f μs\n", 
           atomic_load(&perf_mon.max_spin_ns) / 1000.0);

    fprintf(stderr, "\n[Fallback Strategies]\n");
    fprintf(stderr, "  Zone Rotations: %ld\n", 
           atomic_load(&perf_mon.zone_rotation_count));
    fprintf(stderr, "  Deferred Evictions: %ld\n", 
           atomic_load(&perf_mon.deferred_eviction_count));
    fprintf(stderr, "  Deferred Processed: %ld\n", 
           atomic_load(&perf_mon.deferred_processed_count));
    fprintf(stderr, "  Pending Queue: %d\n", 
           atomic_load(&c->deferred_queue->size));

    fprintf(stderr, "\n[Top 5 Contended Zones]\n");
    long max_zones[5] = {0};
    int max_ids[5] = {-1, -1, -1, -1, -1};

    for (int i = 0; i < 1024; i++) {
        long count = atomic_load(&perf_mon.zone_contention[i]);
        if (count > 0) {
            for (int j = 0; j < 5; j++) {
                if (count > max_zones[j]) {
                    for (int k = 4; k > j; k--) {
                        max_zones[k] = max_zones[k-1];
                        max_ids[k] = max_ids[k-1];
                    }
                    max_zones[j] = count;
                    max_ids[j] = i;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < 5; i++) {
        if (max_ids[i] >= 0) {
            fprintf(stderr, "  Zone %d: %ld contentions\n", max_ids[i], max_zones[i]);
        }
    }
    fprintf(stderr, "\n");
}
