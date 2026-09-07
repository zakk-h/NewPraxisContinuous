// this file is the implementation of our continuous rashomon set methods.
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <memory>
#include <string>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <cstdint>
#include <deque>
#include <map>
#include <span>
#include <chrono>
#include <tuple>
#include <utility>
#include <random>

#include <fstream>

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
  #include <psapi.h>
  #if defined(_MSC_VER)
    #pragma comment(lib, "Psapi.lib")
  #endif
#elif defined(__APPLE__)
  #include <mach/mach.h>
#else
  #include <unistd.h>
#endif

using namespace std;

// fast bitvector computations used throughout. the names are reasonably self-explanatory. popcount of a bitvector, doing bitwise and with a popcount, etc.
#if defined(_MSC_VER)
  #include <intrin.h>
  static inline int popcnt64(uint64_t x) {
      return static_cast<int>(__popcnt64(x));
  }
#else
  static inline int popcnt64(uint64_t x) {
      return __builtin_popcountll(x);
  }
#endif

#if defined(__AVX512F__)
  #include <immintrin.h>
  #define ArborEnum_USE_AVX512 1
#else
  #define ArborEnum_USE_AVX512 0
#endif

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
  #define ArborEnum_USE_AVX512_POPCNT 1
#else
  #define ArborEnum_USE_AVX512_POPCNT 0
#endif

static inline int popcount_words(const uint64_t* a, int n_words) {
#if ArborEnum_USE_AVX512_POPCNT
    int i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i + 8 <= n_words; i += 8) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i pc = _mm512_popcnt_epi64(va);
        acc = _mm512_add_epi64(acc, pc);
    }

    alignas(64) uint64_t tmp[8];
    _mm512_store_si512((void*)tmp, acc);

    uint64_t total =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] +
        tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n_words; ++i) {
        total += (uint64_t)popcnt64(a[i]);
    }

    return (int)total;
#else
    int total = 0;
    for (int i = 0; i < n_words; ++i) {
        total += popcnt64(a[i]);
    }
    return total;
#endif
}

static inline int popcount_and_words(
    const uint64_t* a,
    const uint64_t* b,
    int n_words
) {
#if ArborEnum_USE_AVX512_POPCNT
    int i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i + 8 <= n_words; i += 8) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));
        __m512i vc = _mm512_and_si512(va, vb);
        __m512i pc = _mm512_popcnt_epi64(vc);
        acc = _mm512_add_epi64(acc, pc);
    }

    alignas(64) uint64_t tmp[8];
    _mm512_store_si512((void*)tmp, acc);

    uint64_t total =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] +
        tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n_words; ++i) {
        total += (uint64_t)popcnt64(a[i] & b[i]);
    }

    return (int)total;
#else
    int total = 0;

    for (int i = 0; i < n_words; ++i) {
        total += popcnt64(a[i] & b[i]);
    }

    return total;
#endif
}

static inline int popcount_xor_and_words(
    const uint64_t* mask,
    const uint64_t* a,
    const uint64_t* b,
    int n_words
) {
#if ArborEnum_USE_AVX512_POPCNT
    int i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i + 8 <= n_words; i += 8) {
        __m512i vm = _mm512_loadu_si512((const void*)(mask + i));
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));

        __m512i diff = _mm512_xor_si512(va, vb);
        __m512i active_diff = _mm512_and_si512(vm, diff);
        __m512i pc = _mm512_popcnt_epi64(active_diff);

        acc = _mm512_add_epi64(acc, pc);
    }

    alignas(64) uint64_t tmp[8];
    _mm512_store_si512((void*)tmp, acc);

    uint64_t total =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] +
        tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n_words; ++i) {
        total += (uint64_t)popcnt64(mask[i] & (a[i] ^ b[i]));
    }

    return (int)total;
#else
    int total = 0;
    for (int i = 0; i < n_words; ++i) {
        total += popcnt64(mask[i] & (a[i] ^ b[i]));
    }
    return total;
#endif
}

static inline void and_words(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* out,
    int n_words,
    uint64_t tail_mask
) {
    if (n_words <= 0) return;
#if ArborEnum_USE_AVX512
    int i = 0;
    for (; i + 8 <= n_words; i += 8) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));
        __m512i vc = _mm512_and_si512(va, vb);
        _mm512_storeu_si512((void*)(out + i), vc);
    }
    for (; i < n_words; ++i) out[i] = a[i] & b[i];
#else
    for (int i = 0; i < n_words; ++i) out[i] = a[i] & b[i];
#endif
    out[n_words - 1] &= tail_mask;
}

static inline void andnot_words(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* out,
    int n_words,
    uint64_t tail_mask
) {
    if (n_words <= 0) return;
#if ArborEnum_USE_AVX512
    int i = 0;
    for (; i + 8 <= n_words; i += 8) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));
        __m512i vc = _mm512_andnot_si512(vb, va); // ~b & a
        _mm512_storeu_si512((void*)(out + i), vc);
    }
    for (; i < n_words; ++i) out[i] = a[i] & ~b[i];
#else
    for (int i = 0; i < n_words; ++i) out[i] = a[i] & ~b[i];
#endif
    out[n_words - 1] &= tail_mask;
}

// efficient splitting method
static inline int popcount_and_make_split_words(
    const uint64_t* mask,
    const uint64_t* split,
    uint64_t* left,
    uint64_t* right,
    int n_words,
    uint64_t tail_mask
) {
    if (n_words <= 0) return 0;
#if ArborEnum_USE_AVX512_POPCNT
    int i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i + 8 <= n_words; i += 8) {
        __m512i vm = _mm512_loadu_si512((const void*)(mask + i));
        __m512i vs = _mm512_loadu_si512((const void*)(split + i));

        __m512i vl = _mm512_and_si512(vm, vs);
        __m512i vr = _mm512_andnot_si512(vs, vm); // ~split & mask

        _mm512_storeu_si512((void*)(left + i), vl);
        _mm512_storeu_si512((void*)(right + i), vr);

        __m512i pc = _mm512_popcnt_epi64(vl);
        acc = _mm512_add_epi64(acc, pc);
    }

    alignas(64) uint64_t tmp[8];
    _mm512_store_si512((void*)tmp, acc);

    uint64_t total =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] +
        tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n_words; ++i) {
        const uint64_t l = mask[i] & split[i];
        const uint64_t r = mask[i] & ~split[i];
        left[i] = l;
        right[i] = r;
        total += (uint64_t)popcnt64(l);
    }

    left[n_words - 1] &= tail_mask;
    right[n_words - 1] &= tail_mask;

    return (int)total;
#else
    int total = 0;
    for (int i = 0; i < n_words; ++i) {
        const uint64_t l = mask[i] & split[i];
        const uint64_t r = mask[i] & ~split[i];
        left[i] = l;
        right[i] = r;
        total += popcnt64(l);
    }
    left[n_words - 1] &= tail_mask;
    right[n_words - 1] &= tail_mask;
    return total;
#endif
}

// a low-level packed-bitvector operation used to efficiently carry out claims in the paper
static inline bool any_words(const uint64_t* a, int n_words) {
    if (n_words <= 0) return false;

#if ArborEnum_USE_AVX512
    int i = 0;
    __m512i accum = _mm512_setzero_si512();

    for (; i + 8 <= n_words; i += 8) {
        __m512i v = _mm512_loadu_si512((const void*)(a + i));
        accum = _mm512_or_si512(accum, v);
    }

    if (_mm512_test_epi64_mask(accum, accum) != 0) {
        return true;
    }

    for (; i < n_words; ++i) {
        if (a[i]) return true;
    }

    return false;
#else
    for (int i = 0; i < n_words; ++i) {
        if (a[i]) return true;
    }
    return false;
#endif
}

using Lit = uint32_t; // 32-bit literal = 2*feat + sign
using PathKey = std::vector<Lit>;
static constexpr int DEFER_PREDICTION = -1;

// struct ContinuousPathEntry {
//     int threshold_index = -1; // actual binarized threshold-column index
//     bool went_true = false; // true means left branch: x <= threshold
// };

// using ContinuousPath = std::vector<ContinuousPathEntry>;

// a continuous path entry corresponds to a part of the threshold registry in the paper
// this is just the lower and upper bounds; the active threshold set is maintained separately
struct ContinuousPathEntry {
    // active threshold interval for one continuous feature group.
    // valid thresholds are [lo, hi).
    int lo = -1;
    int hi = -1;
};

using ContinuousPath = std::vector<ContinuousPathEntry>;

static inline const ContinuousPath& empty_continuous_path() {
    static const ContinuousPath p;
    return p;
}

// this is our efficient bitvector representation. a vector of 64-bit words. this is referenced at the top of the appendix, but it also builds on existing work.
struct Packed {
    vector<uint64_t> w; // words (64-bit each)
    Packed() = default;
    explicit Packed(size_t nwords) : w(nwords, 0ULL) {} // allocates a vector of nwords many 64-bit words, with all bits off

    inline void clear() {
        if (!w.empty()) {
            std::memset(w.data(), 0, w.size() * sizeof(uint64_t));
        }
    }

    inline bool any() const {
        return any_words(w.data(), (int)w.size());
    }

    inline int count() const {
        return popcount_words(w.data(), (int)w.size());
    }

};

// scramble a 64-bit value
static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

// hash the array of words into a 64-bit hash value: many-to-one in theory, but with our expected amount of pruning, something like 54k total keys for a reasonably sized rashomon set computation, which yields something like 10^-11 probability of having a collision somewhere.
// https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
static inline uint64_t hash_mask64(const uint64_t* w, int n_words, uint64_t tail_mask) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < n_words; ++i) {
        uint64_t x = w[i];
        if (i == n_words - 1) x &= tail_mask;
        uint64_t m = mix64(x);
        h ^= m + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

struct Key128 {
    uint64_t hi = 0;
    uint64_t lo = 0;

    bool operator==(const Key128& o) const {
        return hi == o.hi && lo == o.lo;
    }

    struct Hash {
        size_t operator()(const Key128& x) const noexcept {
            uint64_t h = x.lo;
            h ^= x.hi + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return static_cast<size_t>(h);
        }
    };
};

// if we want a higher certainty of no collisions, we could use a 128-bit fingerprint instead of a 64 bit fingerprint. this is in addition to what is discussed at the top of the appendix.
static inline Key128 hash_mask128(
    const uint64_t* w,
    int n_words,
    uint64_t tail_mask
) {
    uint64_t h1 = 0x9e3779b97f4a7c15ULL;
    uint64_t h2 = 0xbf58476d1ce4e5b9ULL;

    for (int i = 0; i < n_words; ++i) {
        uint64_t x = w[i];
        if (i == n_words - 1) x &= tail_mask;

        const uint64_t m1 = mix64(x ^ 0x9e3779b97f4a7c15ULL);
        const uint64_t m2 = mix64(x ^ 0xbf58476d1ce4e5b9ULL);

        h1 ^= m1 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2);
        h2 ^= m2 + 0xbf58476d1ce4e5b9ULL + (h2 << 6) + (h2 >> 2);
    }

    return Key128{h1, h2};
}

// creates a canonical cache key to represent the subproblem.
class Fingerprint128IdTable {
public:
    uint64_t intern(Key128 fp) {
        auto it = table.find(fp);
        if (it != table.end()) return it->second;

        const uint64_t id = next_id++;
        table.emplace(fp, id);
        return id;
    }

    size_t size() const {
        return table.size();
    }

private:
    std::unordered_map<Key128, uint64_t, Key128::Hash> table;
    uint64_t next_id = 0;
};

// if we are using literal encoding (existing in this code, not our main contribution)
// 2*feat + sign
static inline Lit enc_lit(int feat, int sign01) {
    if (feat < 0) {
        throw std::runtime_error("enc_lit got negative feature index.");
    }

    const uint64_t lit =
        (static_cast<uint64_t>(feat) << 1) |
        static_cast<uint64_t>(sign01 & 1);

    if (lit > static_cast<uint64_t>(std::numeric_limits<Lit>::max())) {
        throw std::runtime_error("feature index too large for Lit encoding.");
    }

    return static_cast<Lit>(lit);
}

static inline const PathKey& empty_pk() {
    static const PathKey k;
    return k;
}

// existing, not part of our contribution
// insert literal into PathKey, maintaining sorted canonical order
static inline void pk_insert_sorted(PathKey& pk, Lit lit) {
    auto it = std::lower_bound(pk.begin(), pk.end(), lit);
    pk.insert(it, lit);
}

// existing, not part of our contribution
// remove literal from PathKey (must exist)
static inline void pk_erase_sorted(PathKey& pk, Lit lit) {
    auto it = std::lower_bound(pk.begin(), pk.end(), lit);
    pk.erase(it);
}

// efficient way to get multi-class predictions
struct PackedPredMulti {
    std::vector<Packed> by_class; // size = num_classes, each is n_words over eval rows
    Packed deferred; // rows deferred by this tree
};

// bucket by objective
struct ObjBucketMulti {
    int obj;
    std::vector<PackedPredMulti> preds; // one entry per tree at this objective
};

// predictions attached to objective. these methods are useful in extraction or for variable importance.
// not part of our continuous-feature contribution
struct PredPackWithObj {
    int obj;
    PackedPredMulti pred;
};

// used for exact, non-probabilistic keyks at the expense of more memory. we intern the exact bytes of a mask/bitvector and assign a small integer ID.
// first unique mask id 0, second unique mask id 1 and so on.
class MaskIdTable {
public:
    uint32_t intern(const Packed& mask, int n_words, uint64_t tail_mask) {
        const size_t bytes = (size_t)n_words * sizeof(uint64_t); // constant across the dataset, how many words needed * 64 bit length
        string key;
        key.resize(bytes);
        // uint64_t* out = reinterpret_cast<uint64_t*>(&key[0]); // pointer to the start of key
        for (int i = 0; i < n_words; ++i) {
            uint64_t x = mask.w[i];
            if (i == n_words - 1) x &= tail_mask; // the last word may have padding bits, tail_mask zeroes out the unused bits.
            // out[i] = x; // the byte representation of mask.w - we need to convert to use as a key in the unordered map
            std::memcpy(&key[i * sizeof(uint64_t)], &x, sizeof(uint64_t));
        }
        auto it = table.find(key); // have we seen this bitmask before?
        if (it != table.end()) return it->second; // return the previously assigned id if it points to the entry, meaning we have it already
        uint32_t id = (uint32_t)pool_size++; // use the value, then increment it
        table.emplace(std::move(key), id); // store without copying
        return id;
    }

    size_t size() const { return pool_size; }

private:
    unordered_map<string, uint32_t> table;
    size_t pool_size = 0;
};

// construct canonical cache keys (interning) for literals (not part of our contribution)
class LitIdTable {
public:
    uint32_t intern(const std::vector<Lit>& lits) {
        const size_t bytes = lits.size() * sizeof(Lit);
        std::string key;
        key.resize(bytes);
        if (bytes) std::memcpy(&key[0], lits.data(), bytes);

        auto it = table.find(key);
        if (it != table.end()) return it->second;
        uint32_t id = (uint32_t)pool_size++;
        table.emplace(std::move(key), id);
        return id;
    }


    size_t size() const { return pool_size; }

private:
    std::unordered_map<std::string, uint32_t> table;
    size_t pool_size = 0;
};


// two structures to define the key type used in hash maps
// K2: for greedy and lickety cache (subproblem, depth)
// K3: tries (subproblem, depth, budget). In our contribution, we phase out K3 for budget-indepenent subgraph caching - to come later.
// define equality with operator== and how to hash the keys for unordered_map

struct K2 {
    uint64_t k; // hash or interned-id
    int depth;
    bool operator==(const K2& o) const { return k == o.k && depth == o.depth; } // element-wise equality
    struct Hash { // custom hash
        size_t operator()(const K2& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

struct K3 {
    uint64_t k; // hash or interned-id
    int depth;
    int budget;
    bool operator==(const K3& o) const { return k == o.k && depth == o.depth && budget == o.budget; }
    struct Hash {
        size_t operator()(const K3& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            size_t b = (size_t)x.budget;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= b + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

struct KLA {
    uint64_t k;
    int depth;
    int la; // lookahead used for this call
    bool operator==(const KLA& o) const { return k == o.k && depth == o.depth && la == o.la; }
    struct Hash {
        size_t operator()(const KLA& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            size_t a = (size_t)x.la;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= a + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

// this can be used in continuous features if greedy is ever allowed to use more features than licketysplit. then, it is the analog to how we add a licketysplit feature to the enumeration loop in the anytime algorithm.
// we do not perform experiments with this, so it is not part of our contribution.
struct GreedyObjFirstSplit {
    int obj = std::numeric_limits<int>::max();
    int first_feat = -1;
};

//  key used for the threshold proxy completion cache. that is, the map in the main paper.
struct KContProxy {
    uint64_t k; // 64-bit fingerprint or interning id
    int depth;
    int start_idx; // continuous feature that we are working with

    bool operator==(const KContProxy& o) const {
        return k == o.k && depth == o.depth && start_idx == o.start_idx;
    }

    struct Hash {
        size_t operator()(const KContProxy& x) const noexcept {
            size_t h = (size_t)x.k;
            h ^= (size_t)x.depth + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= (size_t)x.start_idx + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
};

using ProxyCompletionTree = std::map<int, std::pair<int,int>>;



// existing data structures for storing the Rashomon set - not part of our contribution.
struct HistEntry {
    int obj;
    uint64_t cnt;
};
static inline bool hist_less(const HistEntry& a, const HistEntry& b){ return a.obj < b.obj; } // helper for sorting

// we preserve the naming in the class. These correspond to OR nodes in our paper.
struct TreeTrieNode; // fwd

struct LeafNode {
    // 0..C-1 means predict that class.
    // DEFER_PREDICTION means defer to the black-box model.
    int prediction;
    int loss;       // gamma + miscls + blackboxpenalty
};

// these correspond to AND nodes in our paper.
struct SplitNode {
    int feature = -1;
    shared_ptr<TreeTrieNode> left;
    shared_ptr<TreeTrieNode> right;
    uint64_t num_valid_trees = 0; // trees contributed by this split under parent's budget
};

struct ExportLeafNode {
    int id = -1;
    int parent_trie_id = -1;
    int prediction = -1;
    int loss = 0;
    int subproblem_size = 0;
};

struct ExportSplitNode {
    int id = -1;
    int parent_trie_id = -1;

    // internal ArborEnum threshold-feature index.
    int feature = -1;

    int left_trie_id = -1;
    int right_trie_id = -1;

    // best completion objective if this split is chosen.
    int min_objective = std::numeric_limits<int>::max();
};

struct ExportTreeTrieNode {
    int id = -1;

    int budget = 0;
    int min_objective = std::numeric_limits<int>::max();
    int subproblem_size = 0;

    std::vector<int> leaf_ids;
    std::vector<int> split_ids;
};

struct ExportANDORGraph {
    int root_trie_id = -1;

    std::vector<ExportTreeTrieNode> trie_nodes;
    std::vector<ExportSplitNode> split_nodes;
    std::vector<ExportLeafNode> leaf_nodes;
};

// this is existing work. we do not claim it as our contribution, though we will utilize some caching later to create budget-independent nodes.
// by budget independent, they still will store a budget, but there will not exist multiple different nodes for the same subproblem and remaining depth, but with different budgets.
// the largest budget which we solve it with is useful metadata, as illustrated in the paper.
struct TreeTrieNode {
    int budget = 0;
    int min_objective = numeric_limits<int>::max();
    vector<LeafNode> leaves; // (prediction,loss) for as many [<=2 in binary classification] if within budget
    vector<SplitNode> splits; // stores splitnodes which have the feature they split on and left and right trienodes
    vector<HistEntry> hist; // sorted ascending by obj; counts aggregated (obj, count) are elements
    bool hist_built = false; // wait until the end to build the histograms because we don't know if they'll be used in the final trie because of multipass

    uint64_t count_trees() const {
        ensure_hist_built();
        uint64_t s = 0;
        for (const auto& e : hist) s += e.cnt;
        return s;
    }

    uint64_t count_leq(int objective) const {
        ensure_hist_built();
        if (hist.empty()) return 0ULL;
        uint64_t total = 0ULL;
        for (const auto& e : hist) {
            if (e.obj > objective) break; // hist is sorted ascending by obj
            total += e.cnt;
        }
        return total;
    }

    void add_hist(int obj, uint64_t add_cnt = 1) {
        auto it = lower_bound(hist.begin(), hist.end(), HistEntry{obj,0}, hist_less); // find the first poisition in hist where obj could be inserted without breaking sort order
        if (it != hist.end() && it->obj == obj) it->cnt += add_cnt; // if it already exists, just increment
        else hist.insert(it, HistEntry{obj, add_cnt}); // otherwise, add
        if (obj < min_objective) min_objective = obj; // keep min_objective fresh
    }
    
    void add_leaf(int prediction, int loss) {
        leaves.push_back(LeafNode{prediction, loss});
        if (loss < min_objective) min_objective = loss;
    }
    
    void add_leaf_and_build(int prediction, int loss) { // assumes you call within budget
        leaves.push_back(LeafNode{prediction, loss});
        add_hist(loss, 1);
    }

    void add_split(int feat,
               const shared_ptr<TreeTrieNode>& L,
               const shared_ptr<TreeTrieNode>& R) {

        if (
            !L ||
            !R ||
            (L->leaves.empty() && L->splits.empty()) ||
            (R->leaves.empty() && R->splits.empty())
        ) {
            return;
        }

        SplitNode s;
        s.feature = feat;
        s.left  = L;
        s.right = R;
        s.num_valid_trees = 0; // will be filled in post-processing

        if (L && R) {
            int min_sum = (L->min_objective == numeric_limits<int>::max() ||
                        R->min_objective == numeric_limits<int>::max())
                        ? numeric_limits<int>::max()
                        : (L->min_objective + R->min_objective);
            if (min_sum < min_objective) min_objective = min_sum;
        }
        splits.push_back(std::move(s));
    }

    void add_split_and_build(int feat,
                   const shared_ptr<TreeTrieNode>& L,
                   const shared_ptr<TreeTrieNode>& R) {
        SplitNode s;
        s.feature = feat;
        s.left  = L;
        s.right = R;

        int min_sum = L->min_objective + R->min_objective;
        if (min_sum < min_objective)
            min_objective = min_sum;

        if ((L && !L->hist.empty()) && (R && !R->hist.empty())) {
            unordered_map<int, uint64_t> sum_counts; // make a temporary map to map obj -> count until we know how they distribute in full to then transfer to the vector-based histogram
            sum_counts.reserve(L->hist.size() * 2); // 2x is a good starting estimate

            uint64_t valid = 0;
            vector<int> R_objs; R_objs.reserve(R->hist.size()); // split (obj, cnt) into two parallel ararys, just for R due to the binary search needs in the future
            vector<uint64_t> R_cnts; R_cnts.reserve(R->hist.size());
            for (auto &e : R->hist) { R_objs.push_back(e.obj); R_cnts.push_back(e.cnt); }

            for (const auto& le : L->hist) {
                if (le.obj > budget) break; // should never happen by invariant but if we do lossy caching/more heuristics
                int rem = budget - le.obj; // R cannot exceed
                auto it_end = upper_bound(R_objs.begin(), R_objs.end(), rem); // find the first index strictly greater than rim
                int idx_end = (int)distance(R_objs.begin(), it_end); // gets the index of it_end (it_end is an iterator)
                for (int j = 0; j < idx_end; ++j) { // go until the last index that doesn't exceed rem
                    int tot = le.obj + R_objs[j];
                    uint64_t addc = le.cnt * R_cnts[j];
                    sum_counts[tot] += addc;
                    valid += addc;
                }
            }
            // we've updated our temporary sum_counts map, now we must merge it into the existing histogram
            if (!sum_counts.empty()) {
                vector<HistEntry> tmp; tmp.reserve(sum_counts.size());
                for (auto &kv : sum_counts) tmp.push_back(HistEntry{kv.first, kv.second}); // back the (obj, count) format and sorting
                sort(tmp.begin(), tmp.end(), hist_less);

                // now, we have to aggregate this into the histogram for all splits at that node
                vector<HistEntry> merged; merged.reserve(hist.size() + tmp.size());
                // simply merge two sorted lists into a new list and swap it in
                size_t i=0, j=0;
                while (i < hist.size() && j < tmp.size()) {
                    if (hist[i].obj < tmp[j].obj) merged.push_back(hist[i++]);
                    else if (tmp[j].obj < hist[i].obj) merged.push_back(tmp[j++]);
                    else { merged.push_back(HistEntry{hist[i].obj, hist[i].cnt + tmp[j].cnt}); ++i; ++j; }
                }
                while (i < hist.size()) merged.push_back(hist[i++]);
                while (j < tmp.size()) merged.push_back(tmp[j++]);
                hist.swap(merged);
            }
            s.num_valid_trees = valid;
        }

        splits.push_back(std::move(s)); // adding this split information to the trienode
    }

    // post-process the trie to build per-node histograms using the existing helpers.
    // assumes leaves/splits/min_objective/budget are already set by construct_trie.
    static void build_histograms_post(TreeTrieNode* node) {
        if (node->hist_built) return;

        // ensure children are processed first (post-order)
        for (auto &s : node->splits) {
            if (s.left)  build_histograms_post(s.left.get());
            if (s.right) build_histograms_post(s.right.get());
        }

        // rebuild this node's histogram from scratch
        std::vector<SplitNode> saved = std::move(node->splits);
        node->splits.clear();
        node->hist.clear();
        node->hist_built = false; // (will set true at end)

        // add leaf contributions
        for (const auto &leaf : node->leaves) {
            node->add_hist(leaf.loss, 1); // could call add_leaf_and_build but that is overkill here
        }

        // re-add splits, letting add_split_and_build do the heavy lifting:
        // merges L/R histograms into node->hist
        // computes s.num_valid_trees
        // refreshes min_objective though that isn't needed
        for (auto &s : saved) {
            node->add_split_and_build(s.feature, s.left, s.right);
        }

        node->hist_built = true;
    }

    void ensure_hist_built() const {
        if (!hist_built) {
            TreeTrieNode::build_histograms_post(const_cast<TreeTrieNode*>(this));
        }
    }

};

struct PredNode {
    int feature;  // -1 for leaf
    int prediction; // only meaningful if feature == -1
    shared_ptr<PredNode> left;
    shared_ptr<PredNode> right;
};

// for joint rashomon set prediction / rid
// struct ObjBucket {
//     int obj;
//     std::vector<Packed> preds; // each is a prediction bitvector for one tree at this obj. predictions for all trees with an objective.
// };

// useful in variable importance. not part of our contribution.
struct EvalCtx {
    int n_eval = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL;
    std::vector<Packed> X_bits_eval; // everything needed for evaluation dataset
};


class ArborEnum {
public:
    // we use hash64 in all of our experiments.
    enum class KeyMode { HASH64, HASH128, EXACT, LITS_EXACT };

    // we ablate these choices in the appendix.
    enum class GreedyContinuousMode {
        BINARY = 0,
        NUMERICAL = 1
    };

    static double current_memory_mb() {
        return current_memory_mb_();
    }


private:
    int n_samples = 0;
    int n_features = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL; // to clear high bits in last word
    int gamma = 0;
    int8_t trained_depth_budget = -1; 

    int best_objective = 0;
    int obj_bound = 0;

    double multiplicative_slack = 0.0;
    bool additive = false;

    vector<Packed> X_bits; // vector of Packed, each Packed is a feature column. packed is a sequence of 64-bit words where each bit corresponds to the row value for the column
    // Packed Ypos; // each bit of a word is the label for the row
    int num_classes = 0;
    std::vector<Packed> Y_bits; // vector is size size num_classes; Y_bits[c] has 1s where y==c
    std::vector<int> continuous_starts;

    bool use_deferral = false;
    double eta_defer = 0.0;

    // training rows where the black-box model is wrong.
    // BBwrong[i] == 1 iff bb_pred[i] != y[i].
    Packed BBwrong;

    bool has_prepared_data = false;

    std::vector<std::vector<bool>> prepared_X_col_major;
    std::vector<int> prepared_y;
    std::vector<int> prepared_continuous_starts;
    std::vector<int> prepared_initial_active_features;
    std::vector<int> prepared_allowed_proxy_features;

    // optional black-box predictions corresponding to prepared_y.
    std::vector<int> prepared_bb_pred;

    KeyMode key_mode = KeyMode::HASH64; // will change later in fit
    GreedyContinuousMode greedy_continuous_mode = GreedyContinuousMode::BINARY;
    bool trie_cache_enabled = false;
    bool proxy_caching_enabled = true;
    mutable MaskIdTable mask_ids; // used only if in exact mode
    mutable LitIdTable lit_ids; // for itemset mode
    mutable Fingerprint128IdTable fingerprint128_ids; // used only in hash128 mode

    int lookahead_init = 1; // will be changed later
    bool use_multipass = true; // sim
    bool rule_list_mode = false;
    bool majority_leaf_only = false;
    bool cache_cheap_subproblems = false;
    bool evaluated_use_min_objectives = false;
    int greedy_split_mode = 1;
    bool stronger_rollout = false;
    // int num_proxy_features = -1; // <=0 means use all feature. positive for feature selection
    
    // empty means unrestricted / all features.
    // non empty means these exact feature indices are allowed when the relevant boolean is on.
    std::vector<int> allowed_proxy_features;
    bool restrict_proxy_in_lickety = false;
    bool restrict_proxy_in_depthd_exact = false;
    bool restrict_proxy_in_greedy = false;
    
    int proxy_style = 0; // 0=constant k (current), 1=cyclic (recursively applying split), 2=cyclic-consistent, 3=split without postprocessing, 4=split with postprocessing
    std::vector<int> k_at_depth; // size depth_budget (only used for style 2)

    unordered_map<K2, int, K2::Hash> greedy_cache;
    unordered_map<K2,  int, K2::Hash>  lickety_cache_k2; // used when lookahead_init <= 1
    unordered_map<KLA, int, KLA::Hash>  lickety_cache_kla; // used when lookahead_init > 1
    // all the ProxyCompletion maps, so this is how we get E in the main paper
    std::unordered_map<
        KContProxy,
        ProxyCompletionTree,
        KContProxy::Hash
    > continuous_proxy_completion_cache;

    unordered_map<K2, GreedyObjFirstSplit, K2::Hash> greedy_first_split_cache;
    unordered_map<K2, int, K2::Hash> anytime_lickety_first_split_cache;
    bool anytime_mode_active_ = false;

    // Optional tao-style refinement of the reconstructed single tree - only in single tree node.
    // when non-null, single-tree prediction and path extraction use this refined tree instead of reconstructing again.
    shared_ptr<PredNode> single_tree_refined_override_;

    // enabled only by fit_then_extend and the explicit anytime algorithm.
    // negative values mean that the corresponding limit is disabled.
    bool resource_limits_active_ = false;
    double runtime_limit_seconds_ = -1.0;
    double memory_limit_mb_ = -1.0;
    std::chrono::steady_clock::time_point resource_start_time_;

    // unordered_map<K3, shared_ptr<TreeTrieNode>, K3::Hash> trie_cache; // if trie_cache_enabled is on
    unordered_map<K2, shared_ptr<TreeTrieNode>, K2::Hash> trie_cache; // canonical node per (subproblem, depth)

    std::vector<std::vector<double>> numerical_X_cols_for_greedy;
    std::vector<std::vector<int>> numerical_global_sorted_idx;
    std::vector<std::vector<double>> numerical_unique_values_for_greedy;
    std::vector<int> y_train;

    // bitwise and of the bitvectors represented as lists of words. makes a sparser list of words
    inline void and_bits(const Packed& a, const Packed& b, Packed& out) const {
        and_words(a.w.data(), b.w.data(), out.w.data(), n_words, tail_mask);
    }

    inline void andnot_bits(const Packed& a, const Packed& b, Packed& out) const {
        andnot_words(a.w.data(), b.w.data(), out.w.data(), n_words, tail_mask);
    }

    inline int popcount_and(const Packed& a, const Packed& b) const {
        return popcount_and_words(a.w.data(), b.w.data(), n_words);
    }

    inline uint64_t key_of_mask(const Packed& mask) const {
        if (key_mode == KeyMode::LITS_EXACT) {
            throw std::runtime_error(
                "key_of_mask called in LITS_EXACT mode; use key_of_state(mask, pk)"
            );
        }
        return key_of_state(mask, empty_pk());
    }

    inline uint64_t key_of_state(const Packed& mask, const PathKey& pk) const {
        switch (key_mode) {
            case KeyMode::HASH64:
                return hash_mask64(mask.w.data(), n_words, tail_mask);

            case KeyMode::HASH128:
                return fingerprint128_ids.intern(
                    hash_mask128(mask.w.data(), n_words, tail_mask)
                );

            case KeyMode::EXACT:
                return static_cast<uint64_t>(
                    mask_ids.intern(mask, n_words, tail_mask)
                );

            case KeyMode::LITS_EXACT:
                return static_cast<uint64_t>(
                    lit_ids.intern(pk)
                );
        }

        return 0;
    }

    // the literal representation depends on the path key, nothing else does.
    inline uint64_t key_of_subproblem(const Packed& mask, const PathKey& pk) const {
        if (key_mode == KeyMode::LITS_EXACT) {
            return key_of_state(mask, pk); // interns pk
        }

        return key_of_mask(mask);
    }

    // inline int proxy_feat_count_() const {
    //     return (num_proxy_features > 0) ? std::min(num_proxy_features, n_features) : n_features;
    // }

    // in our work, we essentially have 3 different types of decision tree algorithms. optimal ones, licketysplit, and greedy.
    // this can differentiate between them
    enum class ProxyLoopKind {
        Lickety,
        DepthDExact,
        Greedy
    };

    // we may also consider any of those 3 being restricted to a small binarization. this checks that for a given kind.
    inline bool should_restrict_proxy_features_(ProxyLoopKind kind) const {
        if (allowed_proxy_features.empty()) return false;

        switch (kind) {
            case ProxyLoopKind::Lickety:
                return restrict_proxy_in_lickety;
            case ProxyLoopKind::DepthDExact:
                return restrict_proxy_in_depthd_exact;
            case ProxyLoopKind::Greedy:
                return restrict_proxy_in_greedy;
        }
        return false;
    }

    inline const std::vector<int>& proxy_features_for_(ProxyLoopKind kind) const {
        static const std::vector<int> empty;
        if (should_restrict_proxy_features_(kind)) return allowed_proxy_features;
        return empty; // empty means use normal 0..n_features-1 loop
    }

    // various ways to derive what features you should give the proxy
    inline bool use_restricted_greedy_proxy_() const {
        return !allowed_proxy_features.empty() && restrict_proxy_in_greedy;
    }

    inline bool use_restricted_depthd_exact_proxy_() const {
        return !allowed_proxy_features.empty() && restrict_proxy_in_depthd_exact;
    }

    inline bool use_restricted_lickety_proxy_() const {
        return !allowed_proxy_features.empty() && restrict_proxy_in_lickety;
    }

    inline bool should_add_continuous_greedy_split_to_binary_lickety_() const {
        return use_restricted_lickety_proxy_()
            && !restrict_proxy_in_greedy
            && greedy_continuous_mode == GreedyContinuousMode::BINARY;
    }

    inline bool should_route_continuous_lickety_depth1_to_binary_greedy_() const {
        return !use_restricted_lickety_proxy_()
            && use_restricted_greedy_proxy_()
            && use_restricted_depthd_exact_proxy_();
    }

    // for the anytime algorithm, if the proxy starts out with all of the features, but the enumeration loop doesn't,
    // then we should add the first split from licketysplit to the enumeration thresholds considered at each subproblem. 
    // this method determines this and the following one gets it.
    inline bool anytime_continuous_lickety_k1_first_split_enabled_() const {
        return anytime_mode_active_
            && lookahead_init == 1
            && proxy_style == 0
            && !use_restricted_lickety_proxy_();
    }

    inline int lookup_anytime_lickety_first_split_(
        const Packed& mask,
        int8_t depth,
        const PathKey& pk
    ) const {
        if (!anytime_continuous_lickety_k1_first_split_enabled_()) {
            return -1;
        }

        const uint64_t k = key_of_subproblem(mask, pk);
        auto it = anytime_lickety_first_split_cache.find(K2{k, depth});

        if (it == anytime_lickety_first_split_cache.end()) {
            return -1;
        }

        return it->second;
    }

    inline bool feature_in_sorted_vector_(const std::vector<int>& xs, int f) const {
        return std::binary_search(xs.begin(), xs.end(), f);
    }

    // a simple way to get the proxy objective when you don't know which proxy you are using
    int proxy_completion_objective_(
        const Packed& mask,
        int8_t depth,
        int8_t k_here,
        const PathKey& pk,
        const ContinuousPath& cpath
    ) {
        if (lookahead_init < 0) {
            return leaf_objective(mask);
        }

        if (lookahead_init == 0) {
            return greedy_proxy_objective_(mask, depth, pk, cpath);
        }

        if (proxy_style == 4) {
            return split_algorithm(mask, depth, k_here, pk);
        }

        return lickety_proxy_objective_(mask, depth, k_here, pk, cpath);
    }

    // this method chooses between the various greedy modes: restricted to a binarization, or fully continuous (but then is your data representation sorted lists of indices or  bitvectors)
    // these are compared in the appendix.
    int greedy_proxy_objective_(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        // if greedy is feature-restricted, always use the existing binary feature loop.
        if (restrict_proxy_in_greedy) {
            return train_greedy(mask, depth_budget, pk);
        }

        if (greedy_continuous_mode == GreedyContinuousMode::BINARY) {
            if (should_add_continuous_greedy_split_to_binary_lickety_()) {
                return train_greedy_continuous_with_first_split_(
                    mask,
                    depth_budget,
                    pk,
                    cpath
                ).obj;
            }

            // scan fully binarized threshold columns by continuous group.
            return train_greedy_continuous(
                mask,
                depth_budget,
                pk,
                cpath
            );
        }

        // use raw numerical sorted lists for continuous features.
        // this entry point will construct what is needed
        return greedy_numerical_entry_point(mask, depth_budget, pk);
    }

    void shift_proxy_strength_down_one_() {
        if (true) {
            // snapshot positive-lookahead entries before mutating the map.
            std::vector<std::pair<KLA, int>> source_entries;
            source_entries.reserve(lickety_cache_kla.size());

            for (const auto& [key, objective] : lickety_cache_kla) {
                if (key.la >= 1) {
                    source_entries.emplace_back(key, objective);
                }
            }

            // remove objectives that are not over continuous features
            greedy_cache.clear();

            for (
                auto it = lickety_cache_kla.begin();
                it != lickety_cache_kla.end();
            ) {
                if (it->first.la == 0) {
                    it = lickety_cache_kla.erase(it);
                } else {
                    ++it;
                }
            }

            // move every positive-lookahead objective down by one because of the greedy base case not being optimal
            for (const auto& [key, objective] : source_entries) {
                const KLA shifted_key{
                    key.k,
                    key.depth,
                    key.la - 1
                };

                auto [shifted_it, inserted] =
                    lickety_cache_kla.emplace(
                        shifted_key,
                        objective
                    );

                if (!inserted) {
                    shifted_it->second = std::min(
                        shifted_it->second,
                        objective
                    );
                }

                // lookahead 1 becomes the new bottom-level completion,
                // so make it visible through the greedy lookup path too.
                if (key.la == 1) {
                    const K2 greedy_key{
                        key.k,
                        key.depth
                    };

                    auto [greedy_it, greedy_inserted] =
                        greedy_cache.emplace(
                            greedy_key,
                            objective
                        );

                    if (!greedy_inserted) {
                        greedy_it->second = std::min(
                            greedy_it->second,
                            objective
                        );
                    }
                }
            }

            return;
        }
        // in anytime, we are always using KLA cache, not K2, so no need for an else case here
       
    }

    // decide which exact solver to use. and exact over what features.
    inline int depthd_exact_proxy_objective_(
        const Packed& mask,
        int8_t depth,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (use_restricted_depthd_exact_proxy_()) {
            return depthd_exact_solver_cached(mask, depth, pk);
        }

        if (depth == 1 && should_add_continuous_greedy_split_to_binary_lickety_()) {
            return depth1_exact_solver_cached_continuous_with_first_split_(
                mask,
                pk,
                cpath
            ).obj;
        }
 
        return depthd_exact_solver_cached_continuous(mask, depth, pk, cpath);
    }

    // decide what licketysplit style algorithm to use, and with what features.
    inline int lickety_proxy_objective_(
        const Packed& mask,
        int8_t depth,
        int8_t k,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (use_restricted_lickety_proxy_()) {
            if (proxy_style == 4) {
                return split_algorithm(mask, depth, k, pk);
            }
            return generalized_lickety_split(mask, depth, k, pk);
        }

        if (proxy_style == 4) {
            return split_algorithm(mask, depth, k, pk);
        }
        return generalized_lickety_split_continuous(mask, depth, k, pk, cpath);
    }

    // do we need a separate lookahead integer in the cache key? it depends on our lookahead.
    inline bool use_kla_cache() const { return lookahead_init > 1; }

    void reserve_caches_mid_() {
        // default max_load_factor.

        greedy_cache.reserve(4'000'000);

        if (use_kla_cache()) {
            lickety_cache_kla.reserve(2'000'000);
        } else {
            lickety_cache_k2.reserve(2'000'000);
        }

        if (trie_cache_enabled) {
            trie_cache.reserve(512);
        }
    }

    static double current_memory_mb_() {
    #if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS_EX counters;
        std::memset(&counters, 0, sizeof(counters));
        counters.cb = sizeof(counters);

        if (!GetProcessMemoryInfo(
                GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                sizeof(counters)
            )) {
            return 0.0;
        }

        return static_cast<double>(counters.WorkingSetSize) /
            (1024.0 * 1024.0);

    #elif defined(__APPLE__)
        mach_task_basic_info info;
        mach_msg_type_number_t count =
            MACH_TASK_BASIC_INFO_COUNT;

        const kern_return_t result = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            reinterpret_cast<task_info_t>(&info),
            &count
        );

        if (result != KERN_SUCCESS) {
            return 0.0;
        }

        return static_cast<double>(info.resident_size) /
            (1024.0 * 1024.0);

    #else
        std::ifstream statm("/proc/self/statm");

        long total_pages = 0;
        long resident_pages = 0;

        if (!(statm >> total_pages >> resident_pages)) {
            return 0.0;
        }

        const long page_size = sysconf(_SC_PAGESIZE);

        if (page_size <= 0) {
            return 0.0;
        }

        return (
            static_cast<double>(resident_pages) *
            static_cast<double>(page_size)
        ) / (1024.0 * 1024.0);
    #endif
    }

    void begin_resource_tracking_(
        double runtime_limit_seconds,
        double memory_limit_mb
    ) {
        if (
            runtime_limit_seconds < 0.0 &&
            memory_limit_mb < 0.0
        ) {
            resource_limits_active_ = false;
            runtime_limit_seconds_ = -1.0;
            memory_limit_mb_ = -1.0;
            return;
        }

        resource_limits_active_ = true;
        runtime_limit_seconds_ = runtime_limit_seconds;
        memory_limit_mb_ = memory_limit_mb;
        resource_start_time_ = std::chrono::steady_clock::now();
    }

    void end_resource_tracking_() {
        resource_limits_active_ = false;
        runtime_limit_seconds_ = -1.0;
        memory_limit_mb_ = -1.0;
    }

    bool resource_limit_reached_() const {
        if (!resource_limits_active_) {
            return false;
        }

        if (runtime_limit_seconds_ >= 0.0) {
            const double elapsed_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    resource_start_time_
                ).count();

            if (elapsed_seconds >= runtime_limit_seconds_) {
                return true;
            }
        }

        if (
            memory_limit_mb_ >= 0.0 &&
            current_memory_mb_() >= memory_limit_mb_
        ) {
            return true;
        }

        return false;
    }

    double elapsed_resource_seconds_() const {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            resource_start_time_
        ).count();
    }

    struct AnytimeRoundCost {
        double seconds = 0.0;
        double memory_mb = 0.0;
    };

    AnytimeRoundCost finish_anytime_round_(
        const std::chrono::steady_clock::time_point& start_time,
        double start_memory_mb
    ) const {
        AnytimeRoundCost cost;

        cost.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time
        ).count();

        cost.memory_mb = std::max(
            0.0,
            current_memory_mb_() - start_memory_mb
        );

        return cost;
    }

    bool auto_proxy_round_is_allowed_(
        const AnytimeRoundCost& previous_round,
        int refinement_round
    ) const {
        const double scale =
            std::ldexp(1.0, refinement_round); // 2^r

        const bool time_ok =
            runtime_limit_seconds_ < 0.0 ||
            previous_round.seconds / runtime_limit_seconds_
                <= scale / 500.0;

        const bool memory_ok =
            memory_limit_mb_ < 0.0 ||
            previous_round.memory_mb / memory_limit_mb_
                <= scale / 37.0;

        return time_ok && memory_ok;
    }
                
    inline int count_total(const Packed& mask) const { return mask.count(); } // number of active samples
    // inline int count_pos(const Packed& mask) const { return popcount_and(mask, Ypos); } // number of active samples that are positive

    inline void count_per_class(const Packed& mask, std::vector<int>& counts) const {
        counts.assign((size_t)num_classes, 0);
        for (int c = 0; c < num_classes; ++c) {
            counts[(size_t)c] = popcount_and(mask, Y_bits[(size_t)c]);
        }
    }

    inline int count_class(const Packed& mask, int c) const {
        return popcount_and(mask, Y_bits[(size_t)c]);
    }

    int rashomon_budget_(
        int reference_objective,
        double rashomon_mult,
        int reference_n,
        bool apply_multiplicative_slack
    ) const {
        double bound = additive
            ? static_cast<double>(reference_objective)
                + rashomon_mult * static_cast<double>(reference_n)
            : static_cast<double>(reference_objective)
                * (1.0 + rashomon_mult);

        if (apply_multiplicative_slack) {
            bound *= (1.0 + multiplicative_slack);
        }

        return static_cast<int>(std::llround(bound));
    }

    // standard formulas in greedy decision tree optimization
    static inline double entropy(double p) {
        const double eps = 1e-12;
        p = max(eps, min(1.0 - eps, p));
        return -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
    }

    static inline double entropy_multiclass(const std::vector<int>& cnts, int n) {
        if (n <= 0) return 0.0;
        const double invn = 1.0 / (double)n;
        double H = 0.0;
        for (int c = 0; c < (int)cnts.size(); ++c) {
            const int k = cnts[(size_t)c];
            if (k <= 0) continue;
            const double p = (double)k * invn;
            H -= p * log2(p);
        }
        return H;
    }

        int canonical_graph_feature_id_(int feat) const {
        if (feat < 0 || feat >= n_features) {
            throw std::runtime_error("canonical_graph_feature_id_ got out-of-range feature.");
        }

        const int first_cont = first_continuous_feature_();

        // ordinary binary feature.
        if (feat < first_cont) {
            return feat;
        }

        // continuous threshold feature: map every threshold in the same
        // continuous group to one canonical id.
        auto it = std::upper_bound(
            continuous_starts.begin(),
            continuous_starts.end(),
            feat
        );

        if (it == continuous_starts.begin()) {
            // should not happen
            return feat;
        }

        const int cont_pos = (int)std::distance(
            continuous_starts.begin(),
            it - 1
        );

        const int group_start = continuous_starts[(size_t)cont_pos];
        const int group_end = continuous_group_end_(cont_pos);

        if (feat >= group_start && feat < group_end) {
            // Put pontinuous-group ids after ordinary binary-feature ids.
            return first_cont + cont_pos;
        }

        // should not happen
        return feat;
    }

    // collecting different features. we use this to evaluate the anytime algorithm over time.
    void collect_graph_feature_ids_(
        const std::shared_ptr<TreeTrieNode>& node,
        std::unordered_set<int>& seen
    ) const {
        if (!node) return;

        for (const auto& s : node->splits) {
            seen.insert(canonical_graph_feature_id_(s.feature));

            collect_graph_feature_ids_(s.left, seen);
            collect_graph_feature_ids_(s.right, seen);
        }
    }

    struct ExactMaskDepthKey {
        std::string bytes;
        int depth;

        bool operator==(const ExactMaskDepthKey& o) const {
            return depth == o.depth && bytes == o.bytes;
        }

        struct Hash {
            size_t operator()(const ExactMaskDepthKey& x) const noexcept {
                size_t h = std::hash<std::string>{}(x.bytes);
                size_t d = (size_t)x.depth;
                h ^= d + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                return h;
            }
        };
    };

    // bitvector cache key
    ExactMaskDepthKey exact_mask_depth_key_(
        const Packed& mask,
        int depth
    ) const {
        ExactMaskDepthKey key;
        key.depth = depth;

        const size_t bytes = (size_t)n_words * sizeof(uint64_t);
        key.bytes.resize(bytes);

        for (int i = 0; i < n_words; ++i) {
            uint64_t x = mask.w[(size_t)i];
            if (i == n_words - 1) {
                x &= tail_mask;
            }

            std::memcpy(
                &key.bytes[(size_t)i * sizeof(uint64_t)],
                &x,
                sizeof(uint64_t)
            );
        }

        return key;
    }

    // this method is used to get the various statistics about the graph so we can understand how compact the different options are
    // trie representation, budget-dependent graph, or budget-independent graph
    // the following few methods handle these cases.
    void collect_graph_subproblem_depth_pairs_(
        const std::shared_ptr<TreeTrieNode>& node,
        const Packed& mask,
        int depth,
        std::unordered_set<
            ExactMaskDepthKey,
            ExactMaskDepthKey::Hash
        >& seen
    ) const {
        if (!node) return;

        seen.insert(exact_mask_depth_key_(mask, depth));

        if (depth <= 0) return;

        Packed L(n_words), R(n_words);

        for (const auto& s : node->splits) {
            and_bits(mask, X_bits[(size_t)s.feature], L);
            andnot_bits(mask, X_bits[(size_t)s.feature], R);

            collect_graph_subproblem_depth_pairs_(
                s.left,
                L,
                depth - 1,
                seen
            );

            collect_graph_subproblem_depth_pairs_(
                s.right,
                R,
                depth - 1,
                seen
            );
        }
    }

        void collect_distinct_or_nodes_(
            const std::shared_ptr<TreeTrieNode>& node,
            std::unordered_set<const TreeTrieNode*>& seen
        ) const {
            if (!node) return;

            const TreeTrieNode* ptr = node.get();

            // already counted this OR node through another parent/path.
            if (seen.find(ptr) != seen.end()) {
                return;
            }

            // a TreeTrieNode is an OR node: it stores leaf options and split options.
            if (!node->leaves.empty() || !node->splits.empty()) {
                seen.insert(ptr);
            }

            for (const auto& s : node->splits) {
                collect_distinct_or_nodes_(s.left, seen);
                collect_distinct_or_nodes_(s.right, seen);
            }
        }

    Packed root_mask_() const {
        Packed root(n_words);

        if (n_words <= 0) {
            return root;
        }

        for (int i = 0; i < n_words - 1; ++i) {
            root.w[(size_t)i] = ~0ULL;
        }

        root.w[(size_t)(n_words - 1)] = tail_mask;
        return root;
    }


    // count the distinct subproblems/bitvectors/literals/fingerprints, not considering depth or greedy/lickety
    size_t count_distinct_subproblems_union() const {
        std::unordered_set<uint64_t> seen;

        const size_t lsz = use_kla_cache() ? lickety_cache_kla.size() : lickety_cache_k2.size();
        const size_t approx = greedy_cache.size() + lsz;
        seen.reserve(approx * 2 + 16);

        // greedy: key is (k, depth). we ignore depth by inserting only k.
        for (const auto& kv : greedy_cache) {
            seen.insert(kv.first.k);
        }

        // lickety: either (k, depth) or (k, depth, la). ignore depth/la by inserting only k.
        if (use_kla_cache()) {
            for (const auto& kv : lickety_cache_kla) {
                seen.insert(kv.first.k);
            }
        } else {
            for (const auto& kv : lickety_cache_k2) {
                seen.insert(kv.first.k);
            }
        }

        return seen.size();
    }

    // what thresholds we consider for continuous features. if max_number_thresholds_per_feature is positive, then this isn't guaranteed to be complete.
    // we evaluate with -1 and 100 in the paper.
    std::vector<double> threshold_values_for_numeric_feature_(
        const std::vector<double>& raw_values,
        const std::vector<double>& sorted_unique_vals,
        int max_number_thresholds_per_feature
    ) const {
        if (sorted_unique_vals.size() <= 1) {
            return {};
        }

        const int candidate_count =
            static_cast<int>(sorted_unique_vals.size()) - 1;

        // exhaustive default: every unique value except max.
        if (
            max_number_thresholds_per_feature <= 0 ||
            candidate_count <= max_number_thresholds_per_feature
        ) {
            return std::vector<double>(
                sorted_unique_vals.begin(),
                sorted_unique_vals.end() - 1
            );
        }

        std::vector<double> sorted_vals;
        sorted_vals.reserve(raw_values.size());

        for (double x : raw_values) {
            if (std::isfinite(x)) {
                sorted_vals.push_back(x);
            }
        }

        if (sorted_vals.empty()) {
            return {};
        }

        std::sort(sorted_vals.begin(), sorted_vals.end());

        std::vector<double> thresholds;
        thresholds.reserve(
            static_cast<std::size_t>(max_number_thresholds_per_feature)
        );

        const int n = static_cast<int>(sorted_vals.size());

        // probabilities 1/(m+1), ..., m/(m+1), with linear interpolation.
        for (int j = 1; j <= max_number_thresholds_per_feature; ++j) {
            const double q =
                static_cast<double>(j) /
                static_cast<double>(max_number_thresholds_per_feature + 1);

            const double pos = q * static_cast<double>(n - 1);
            int lo = static_cast<int>(std::floor(pos));
            int hi = static_cast<int>(std::ceil(pos));

            if (lo < 0) lo = 0;
            if (hi < 0) hi = 0;
            if (lo >= n) lo = n - 1;
            if (hi >= n) hi = n - 1;

            const double frac = pos - static_cast<double>(lo);
            const double thr =
                sorted_vals[(std::size_t)lo] * (1.0 - frac) +
                sorted_vals[(std::size_t)hi] * frac;

            if (std::isfinite(thr) &&
                thr >= sorted_unique_vals.front() &&
                thr < sorted_unique_vals.back()) {
                thresholds.push_back(thr);
            }
        }

        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(
            std::unique(thresholds.begin(), thresholds.end()),
            thresholds.end()
        );

        return thresholds;
    }


public:
    shared_ptr<TreeTrieNode> result;
    void set_key_mode(KeyMode m) { key_mode = m; }
    void set_fingerprint_bits(int bits) { key_mode = (bits == 128) ? KeyMode::HASH128 : KeyMode::HASH64; }
    void set_use_128bit_fingerprint(bool on) { key_mode = on ? KeyMode::HASH128 : KeyMode::HASH64; }
    void set_trie_cache_enabled(bool on) { trie_cache_enabled = on; }
    void set_multiplicative_slack(double s) { multiplicative_slack = s; }
    void set_additive(bool on) { additive = on; }
    void set_use_multipass(bool on) { use_multipass = on; }
    void set_rule_list_mode(bool on) { rule_list_mode = on; }
    void set_majority_leaf_only(bool on) { majority_leaf_only = on; }
    void set_cache_cheap_subproblems(bool on) { cache_cheap_subproblems = on; }
    void set_greedy_split_mode(int m) { greedy_split_mode = m; }
    void set_proxy_caching_enabled(bool on) { proxy_caching_enabled = on; }
    void set_evaluated_use_min_objectives(bool on) { evaluated_use_min_objectives = on; }
    void set_stronger_rollout(bool on) { stronger_rollout = on; }

    void set_allowed_proxy_features(const std::vector<int>& feats) {
        allowed_proxy_features.clear();
        allowed_proxy_features.reserve(feats.size());

        for (int f : feats) {
            if (f < 0 || f >= n_features) {
                throw std::runtime_error("allowed_proxy_features contains out-of-range feature index.");
            }
            allowed_proxy_features.push_back(f);
        }

        std::sort(allowed_proxy_features.begin(), allowed_proxy_features.end());
        allowed_proxy_features.erase(
            std::unique(allowed_proxy_features.begin(), allowed_proxy_features.end()),
            allowed_proxy_features.end()
        );
    }

    void set_proxy_feature_restrictions(
        bool use_in_lickety,
        bool use_in_depthd_exact,
        bool use_in_greedy
    ) {
        restrict_proxy_in_lickety = use_in_lickety;
        restrict_proxy_in_depthd_exact = use_in_depthd_exact;
        restrict_proxy_in_greedy = use_in_greedy;
    }

    void set_greedy_continuous_mode(int mode) {
        if (mode == 0) {
            greedy_continuous_mode = GreedyContinuousMode::BINARY;
        } else if (mode == 1) {
            greedy_continuous_mode = GreedyContinuousMode::NUMERICAL;
        } else {
            throw std::runtime_error(
                "set_greedy_continuous_mode expects 0 for BINARY or 1 for NUMERICAL."
            );
        }

        greedy_cache.clear();
        greedy_first_split_cache.clear();
    }

    int get_greedy_continuous_mode() const {
        return greedy_continuous_mode == GreedyContinuousMode::BINARY ? 0 : 1;
    }

    struct MistakesWithObj {
        int obj;
        int mistakes;
    };

    struct DeferralsWithObj {
        int obj;
        int deferrals;
    };

    struct ExactReplacementMistakesWithObj {
        int obj = 0;
        std::vector<double> mistakes; // [m_original, m_0, ..., m_{p-1}], exact, no montecarlo error
    };

    using ExactImportanceInterval = std::pair<double, double>;

    ExportANDORGraph export_andor_graph(
        std::size_t max_trie_nodes = 10000000,
        std::size_t max_split_nodes = 50000000,
        std::size_t max_leaf_nodes = 50000000
    ) const {
        if (!result) {
            throw std::runtime_error(
                "No Rashomon graph exists. Fit ArborEnum first."
            );
        }

        ExportANDORGraph out;

        // one exported id per actual TreeTrieNode object.
        // this preserves DAG sharing: if several paths point to the same
        // node, they all receive the same exported id.
        std::unordered_map<const TreeTrieNode*, int> trie_ids;

        auto get_trie_id =
            [&](const std::shared_ptr<TreeTrieNode>& node) -> int {
                if (!node) return -1;

                const TreeTrieNode* ptr = node.get();

                auto it = trie_ids.find(ptr);
                if (it != trie_ids.end()) {
                    return it->second;
                }

                if (out.trie_nodes.size() >= max_trie_nodes) {
                    throw std::runtime_error(
                        "export_andor_graph exceeded max_trie_nodes."
                    );
                }

                const int id =
                    static_cast<int>(out.trie_nodes.size());

                trie_ids.emplace(ptr, id);

                ExportTreeTrieNode exported;
                exported.id = id;

                // the rest is filled when this node is processed.
                out.trie_nodes.push_back(std::move(exported));

                return id;
            };

        struct StackItem {
            std::shared_ptr<TreeTrieNode> node;
            Packed mask;
        };

        // root training mask.
        Packed root_mask(static_cast<std::size_t>(n_words));

        for (int w = 0; w < n_words - 1; ++w) {
            root_mask.w[static_cast<std::size_t>(w)] = ~0ULL;
        }

        root_mask.w[static_cast<std::size_t>(n_words - 1)] =
            tail_mask;

        out.root_trie_id = get_trie_id(result);

        std::vector<StackItem> stack;
        stack.push_back(
            StackItem{
                result,
                std::move(root_mask)
            }
        );

        std::unordered_set<const TreeTrieNode*> processed;

        while (!stack.empty()) {
            StackItem item = std::move(stack.back());
            stack.pop_back();

            const auto& cur = item.node;
            if (!cur) continue;

            const TreeTrieNode* ptr = cur.get();

            // prevents a shared DAG node from being exported twice.
            if (!processed.insert(ptr).second) {
                continue;
            }

            const int cur_id = get_trie_id(cur);
            const int cur_size = item.mask.count();

            // do not hold a reference into out.trie_nodes here (had dangling pointer issue)
            out.trie_nodes[
                static_cast<std::size_t>(cur_id)
            ].budget = cur->budget;

            out.trie_nodes[
                static_cast<std::size_t>(cur_id)
            ].min_objective = cur->min_objective;

            out.trie_nodes[
                static_cast<std::size_t>(cur_id)
            ].subproblem_size = cur_size;

            // leaf alternatives
            for (const LeafNode& leaf : cur->leaves) {
                if (out.leaf_nodes.size() >= max_leaf_nodes) {
                    throw std::runtime_error(
                        "export_andor_graph exceeded max_leaf_nodes."
                    );
                }

                ExportLeafNode e;

                e.id =
                    static_cast<int>(out.leaf_nodes.size());

                e.parent_trie_id = cur_id;
                e.prediction = leaf.prediction;
                e.loss = leaf.loss;
                e.subproblem_size = cur_size;

                out.trie_nodes[
                    static_cast<std::size_t>(cur_id)
                ].leaf_ids.push_back(e.id);

                out.leaf_nodes.push_back(std::move(e));
            }

            // split / AND alternatives
            for (const SplitNode& split : cur->splits) {
                if (!split.left || !split.right) {
                    continue;
                }

                if (
                    split.feature < 0 ||
                    split.feature >= static_cast<int>(X_bits.size())
                ) {
                    throw std::runtime_error(
                        "export_andor_graph found an invalid internal feature."
                    );
                }

                if (out.split_nodes.size() >= max_split_nodes) {
                    throw std::runtime_error(
                        "export_andor_graph exceeded max_split_nodes."
                    );
                }

                Packed left_mask(
                    static_cast<std::size_t>(n_words)
                );

                Packed right_mask(
                    static_cast<std::size_t>(n_words)
                );

                and_words(
                    item.mask.w.data(),
                    X_bits[
                        static_cast<std::size_t>(split.feature)
                    ].w.data(),
                    left_mask.w.data(),
                    n_words,
                    tail_mask
                );

                andnot_words(
                    item.mask.w.data(),
                    X_bits[
                        static_cast<std::size_t>(split.feature)
                    ].w.data(),
                    right_mask.w.data(),
                    n_words,
                    tail_mask
                );

                // Ttese may grow out.trie_nodes and reallocate it.
                const int left_id =
                    get_trie_id(split.left);

                const int right_id =
                    get_trie_id(split.right);

                ExportSplitNode e;

                e.id =
                    static_cast<int>(out.split_nodes.size());

                e.parent_trie_id = cur_id;

                // keep the actual internal threshold-feature index.
                e.feature = split.feature;

                e.left_trie_id = left_id;
                e.right_trie_id = right_id;

                if (
                    split.left->min_objective !=
                        std::numeric_limits<int>::max()
                    &&
                    split.right->min_objective !=
                        std::numeric_limits<int>::max()
                ) {
                    e.min_objective =
                        split.left->min_objective +
                        split.right->min_objective;
                }

                // re-index after get_trie_id() calls instead of using
                // a reference that may have been invalidated.
                out.trie_nodes[
                    static_cast<std::size_t>(cur_id)
                ].split_ids.push_back(e.id);

                out.split_nodes.push_back(std::move(e));

                if (
                    processed.find(split.left.get()) ==
                    processed.end()
                ) {
                    stack.push_back(
                        StackItem{
                            split.left,
                            std::move(left_mask)
                        }
                    );
                }

                if (
                    processed.find(split.right.get()) ==
                    processed.end()
                ) {
                    stack.push_back(
                        StackItem{
                            split.right,
                            std::move(right_mask)
                        }
                    );
                }
            }
        }

        return out;
    }

    


    // return the internal feature index at which each continuous group begins
    std::vector<int> get_continuous_starts() const {
        if (has_prepared_data) {
            return prepared_continuous_starts;
        }

        return continuous_starts;
    }

    // return the number of continuous feature groups.
    int get_num_continuous_groups() const {
        if (has_prepared_data) {
            return static_cast<int>(
                prepared_continuous_starts.size()
            );
        }

        return static_cast<int>(continuous_starts.size());
    }

    // return the exclusive end index of a continuous feature group
    int get_continuous_group_end(int continuous_group) const {
        const std::vector<int>& starts =
            has_prepared_data
                ? prepared_continuous_starts
                : continuous_starts;

        if (
            continuous_group < 0 ||
            continuous_group >= static_cast<int>(starts.size())
        ) {
            throw std::runtime_error(
                "continuous_group is out of range."
            );
        }

        if (continuous_group + 1 < static_cast<int>(starts.size())) {
            return starts[
                static_cast<std::size_t>(continuous_group + 1)
            ];
        }

        if (has_prepared_data) {
            return static_cast<int>(
                prepared_X_col_major.size()
            );
        }

        return n_features;
    }

    // return the actual <= cutpoints for one continuous group.
    std::vector<double> get_continuous_cutpoints(
        int continuous_group
    ) const {
        if (
            continuous_group < 0 ||
            continuous_group >= static_cast<int>(
                numerical_unique_values_for_greedy.size()
            )
        ) {
            throw std::runtime_error(
                "continuous_group is out of range or threshold metadata "
                "is unavailable."
            );
        }

        const auto& stored_values =
            numerical_unique_values_for_greedy[
                static_cast<std::size_t>(continuous_group)
            ];

        if (stored_values.size() <= 1) {
            return {};
        }

        return std::vector<double>(
            stored_values.begin(),
            stored_values.end() - 1
        );
    }

    // given an internal binarized feature index, return:
    // continuous_group, offset within that group, cutpoint (as in <=cutpoint)
  
    std::tuple<int, int, double>
    get_continuous_threshold_info(
        int internal_feature
    ) const {
        const std::vector<int>& starts =
            has_prepared_data
                ? prepared_continuous_starts
                : continuous_starts;

        const int total_features =
            has_prepared_data
                ? static_cast<int>(prepared_X_col_major.size())
                : n_features;

        if (
            internal_feature < 0 ||
            internal_feature >= total_features
        ) {
            throw std::runtime_error(
                "internal_feature is out of range."
            );
        }

        if (
            starts.empty() ||
            internal_feature < starts.front()
        ) {
            throw std::runtime_error(
                "internal_feature is an ordinary binary feature, "
                "not a continuous threshold feature."
            );
        }

        auto upper = std::upper_bound(
            starts.begin(),
            starts.end(),
            internal_feature
        );

        const int continuous_group =
            static_cast<int>(
                std::distance(starts.begin(), upper)
            ) - 1;

        if (continuous_group < 0) {
            throw std::logic_error(
                "Failed to locate continuous group."
            );
        }

        const int start =
            starts[static_cast<std::size_t>(continuous_group)];

        const int end =
            continuous_group + 1 < static_cast<int>(starts.size())
                ? starts[
                    static_cast<std::size_t>(continuous_group + 1)
                ]
                : total_features;

        if (internal_feature < start || internal_feature >= end) {
            throw std::logic_error(
                "Internal feature does not fall inside its inferred "
                "continuous group."
            );
        }

        const int offset = internal_feature - start;

        const auto cutpoints =
            get_continuous_cutpoints(continuous_group);

        if (offset < 0 || offset >= static_cast<int>(cutpoints.size())) {
            throw std::logic_error(
                "Continuous feature offset does not match stored cutpoints."
            );
        }

        return std::make_tuple(
            continuous_group,
            offset,
            cutpoints[static_cast<std::size_t>(offset)]
        );
    }

    // encode one numerical value using every <= cutpoint in a continuous group.
    // for cutpoints [1.0, 2.0, 3.0]
    // value 2.0 produces [0, 1, 1]
    std::vector<uint8_t> encode_continuous_value(
        int continuous_group,
        double value
    ) const {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "value must be finite."
            );
        }

        const auto cutpoints =
            get_continuous_cutpoints(continuous_group);

        std::vector<uint8_t> encoded;
        encoded.reserve(cutpoints.size());

        for (double cutpoint : cutpoints) {
            encoded.push_back(
                value <= cutpoint
                    ? static_cast<uint8_t>(1)
                    : static_cast<uint8_t>(0)
            );
        }

        return encoded;
    }

    // return: is_continuous, continuous_group, offset, cutpoint
    // for an ordinary binary feature: (false, -1, -1, NaN)
    std::tuple<bool, int, int, double>
    get_internal_feature_info(
        int internal_feature
    ) const {
        const std::vector<int>& starts =
            has_prepared_data
                ? prepared_continuous_starts
                : continuous_starts;

        const int total_features =
            has_prepared_data
                ? static_cast<int>(prepared_X_col_major.size())
                : n_features;

        if (
            internal_feature < 0 ||
            internal_feature >= total_features
        ) {
            throw std::runtime_error(
                "internal_feature is out of range."
            );
        }

        if (
            starts.empty() ||
            internal_feature < starts.front()
        ) {
            return std::make_tuple(
                false,
                -1,
                -1,
                std::numeric_limits<double>::quiet_NaN()
            );
        }

        const auto [group, offset, cutpoint] =
            get_continuous_threshold_info(internal_feature);

        return std::make_tuple(
            true,
            group,
            offset,
            cutpoint
        );
    }





    // basic graph statistics for the anytime algorithm or for compact storage evaluation
    size_t count_graph_features() const {
        if (!result) {
            return 0;
        }

        std::unordered_set<int> seen;
        seen.reserve(128);

        collect_graph_feature_ids_(result, seen);

        return seen.size();
    }

    size_t count_graph_subproblem_depth_pairs() const {
        if (!result) {
            return 0;
        }

        Packed root = root_mask_();

        std::unordered_set<
            ExactMaskDepthKey,
            ExactMaskDepthKey::Hash
        > seen;

        seen.reserve(1024);

        collect_graph_subproblem_depth_pairs_(
            result,
            root,
            trained_depth_budget,
            seen
        );

        return seen.size();
    }

    size_t count_distinct_or_nodes() const {
        if (!result) {
            return 0;
        }

        std::unordered_set<const TreeTrieNode*> seen;
        seen.reserve(1024);

        collect_distinct_or_nodes_(result, seen);

        return seen.size();
    }

    // this method is new and its goal is to convert raw binary and numerical inputs into packed binary columns, and contiguous continuous-threshold groups. it also will choose what thresholds the proxy uses. we map proxy thresholds to the nearest in hamming distance. it should be 0 if we take all thresholds.
    // this corresponds to how we handle continuous features in the main paper.
    void prepare_continuous_data(
        const std::vector<std::vector<double>>& X_num_row_major,
        const std::vector<std::vector<uint8_t>>& X_bin_row_major,
        const std::vector<int>& y,
        const std::vector<std::vector<uint8_t>>& X_initial_active_row_major,
        const std::vector<std::vector<uint8_t>>& X_proxy_active_row_major,
        int max_number_thresholds_per_feature = -1,
        const std::vector<int>& bb_pred = {}
    ) {
        validate_rectangular_matrix_(X_num_row_major, "X_num_row_major");
        validate_rectangular_matrix_(X_bin_row_major, "X_bin_row_major");
        validate_rectangular_matrix_(X_initial_active_row_major, "X_initial_active_row_major");
        validate_rectangular_matrix_(X_proxy_active_row_major, "X_proxy_active_row_major");
        const int n_num = (int)X_num_row_major.size();
        const int n_bin = (int)X_bin_row_major.size();
        const int n_initial_active = (int)X_initial_active_row_major.size();
        const int n_proxy_active = (int)X_proxy_active_row_major.size();

        int n = -1;

        if (n_num > 0) n = n_num;
        if (n_bin > 0) {
            if (n < 0) n = n_bin;
            else if (n_bin != n) throw std::runtime_error("X_bin_row_major row count does not match X_num_row_major.");
        }
        if (n_initial_active > 0) {
            if (n < 0) {
                n = n_initial_active;
            } else if (n_initial_active != n) {
                throw std::runtime_error(
                    "X_initial_active_row_major row count mismatch."
                );
            }
        }

        if (n_proxy_active > 0) {
            if (n < 0) {
                n = n_proxy_active;
            } else if (n_proxy_active != n) {
                throw std::runtime_error(
                    "X_proxy_active_row_major row count mismatch."
                );
            }
        }

        if (n < 0) {
            throw std::runtime_error("At least one of X_num_row_major, X_bin_row_major, or X_initial_active_row_major or X_proxy_active_row_major must be nonempty.");
        }

        if ((int)y.size() != n) {
            throw std::runtime_error("y length does not match number of rows.");
        }

        const int p_num = (n_num > 0) ? (int)X_num_row_major[0].size() : 0;
        const int p_bin = (n_bin > 0) ? (int)X_bin_row_major[0].size() : 0;

        std::vector<std::vector<bool>> X_cols;
        std::vector<int> cont_starts;
        std::vector<std::vector<double>> numerical_cols_for_greedy;
        std::vector<std::vector<int>> numerical_global_sorted_idx_local;
        std::vector<std::vector<double>> numerical_unique_values_for_greedy_local;

        // first append already-binary columns.
        // these are ordinary binary features, not continuous threshold groups.
        X_cols.reserve((std::size_t)(p_bin + p_num * 8));

        for (int f = 0; f < p_bin; ++f) {
            std::vector<bool> col((std::size_t)n, false);

            for (int i = 0; i < n; ++i) {
                const uint8_t v = X_bin_row_major[(std::size_t)i][(std::size_t)f];
                col[(std::size_t)i] = (v != 0);
            }

            X_cols.push_back(std::move(col));
        }

        // fully threshold-binarize each numeric column.
        // each numeric feature becomes one contiguous continuous threshold group.
        for (int f = 0; f < p_num; ++f) {
            std::vector<double> vals = sorted_unique_values_(X_num_row_major, f);

            // constant numeric feature contributes no threshold columns.
            if (vals.size() <= 1) continue;

            // store the raw numerical column corresponding to this continuous group.
            std::vector<double> col_num((std::size_t)n);
            for (int i = 0; i < n; ++i) {
                col_num[(std::size_t)i] =
                    X_num_row_major[(std::size_t)i][(std::size_t)f];
            }

            std::vector<double> threshold_vals =
                threshold_values_for_numeric_feature_(
                    col_num,
                    vals,
                    max_number_thresholds_per_feature
                );

            if (threshold_vals.empty()) continue;

            const int group_start = (int)X_cols.size();
            cont_starts.push_back(group_start);

            // numerical_unique_values_for_greedy_local must stay aligned with
            // the actual threshold columns. Each value except the last
            // corresponds to one threshold column; the last is a sentinel max.
            std::vector<double> stored_vals = threshold_vals;
            stored_vals.push_back(vals.back());
            numerical_unique_values_for_greedy_local.push_back(std::move(stored_vals));

            // store globally sorted row indices for this numerical feature.
            std::vector<int> order((std::size_t)n);
            for (int i = 0; i < n; ++i) order[(std::size_t)i] = i;

            std::stable_sort(
                order.begin(),
                order.end(),
                [&](int a, int b) {
                    const double xa = col_num[(std::size_t)a];
                    const double xb = col_num[(std::size_t)b];
                    if (xa < xb) return true;
                    if (xb < xa) return false;
                    return a < b;
                }
            );

            numerical_cols_for_greedy.push_back(std::move(col_num));
            numerical_global_sorted_idx_local.push_back(std::move(order));

            // use either exhaustive thresholds or quantile-spaced thresholds,
            // depending on max_number_thresholds_per_feature.
            for (double thr : threshold_vals) {
                std::vector<bool> col((std::size_t)n, false);
                for (int i = 0; i < n; ++i) {
                    col[(std::size_t)i] =
                        X_num_row_major[(std::size_t)i][(std::size_t)f] <= thr;
                }

                X_cols.push_back(std::move(col));
            }
        }

        if (X_cols.empty()) {
            throw std::runtime_error("prepare_continuous_data produced zero binary features.");
        }

        // map active binary columns to closest columns in the full binarized X.
        // for (int a = 0; a < p_active; ++a) {
        //     std::vector<uint8_t> active_col((std::size_t)n, 0);

        //     for (int i = 0; i < n; ++i) {
        //         active_col[(std::size_t)i] =
        //             X_active_row_major[(std::size_t)i][(std::size_t)a] ? 1 : 0;
        //     }

        //     int best_idx = -1;
        //     int best_dist = std::numeric_limits<int>::max();

        //     for (int f = 0; f < (int)X_cols.size(); ++f) {
        //         const int d = hamming_distance_binary_column_(
        //             X_cols[(std::size_t)f],
        //             active_col
        //         );

        //         if (d < best_dist) {
        //             best_dist = d;
        //             best_idx = f;

        //             // Perfect match. Cannot improve.
        //             if (best_dist == 0) break;
        //         }
        //     }

        //     if (best_idx < 0) {
        //         throw std::runtime_error("Failed to map active binary feature to full binarized feature.");
        //     }

        //     active_features.push_back(best_idx);
        // }

        auto map_columns_to_full_features =
            [&](
                const std::vector<std::vector<uint8_t>>& X_selected,
                int first_candidate_feature
            ) -> std::vector<int>
        {
            const int n_selected = (int)X_selected.size();
            const int p_selected =
                n_selected > 0
                    ? (int)X_selected[0].size()
                    : 0;

            std::vector<int> selected_features;
            selected_features.reserve((std::size_t)p_selected);

            for (int a = 0; a < p_selected; ++a) {
                std::vector<uint8_t> selected_col((std::size_t)n, 0);

                for (int i = 0; i < n; ++i) {
                    selected_col[(std::size_t)i] =
                        X_selected[(std::size_t)i][(std::size_t)a]
                            ? 1
                            : 0;
                }

                int best_idx = -1;
                int best_dist = std::numeric_limits<int>::max();

                for (
                    int f = first_candidate_feature;
                    f < (int)X_cols.size();
                    ++f
                ) {
                    const int d_same = hamming_distance_binary_column_(
                        X_cols[(std::size_t)f],
                        selected_col
                    );

                    const int d = std::min(
                        d_same,
                        n - d_same
                    );

                    if (d < best_dist) {
                        best_dist = d;
                        best_idx = f;

                        if (best_dist == 0) {
                            break;
                        }
                    }
                }

                if (best_idx < 0) {
                    throw std::runtime_error(
                        "Failed to map selected binary feature "
                        "to a candidate full feature."
                    );
                }

                selected_features.push_back(best_idx);
            }

            sort_unique_ints_inplace_(selected_features);
            return selected_features;
        };

        const int first_continuous_feature =
            cont_starts.empty()
                ? static_cast<int>(X_cols.size())
                : cont_starts.front();

        std::vector<int> initial_active_features =
            map_columns_to_full_features(
                X_initial_active_row_major,
                first_continuous_feature
            );

        std::vector<int> proxy_active_features =
            map_columns_to_full_features(
                X_proxy_active_row_major,
                0
            );

        prepared_X_col_major = std::move(X_cols);
        prepared_y = y;
        prepared_continuous_starts = std::move(cont_starts);

        prepared_bb_pred = bb_pred;
        if (
            !prepared_bb_pred.empty() &&
            prepared_bb_pred.size() !=
                prepared_y.size()
        ) {
            throw std::runtime_error(
                "Prepared bb_pred must have the same "
                "length as prepared y."
            );
        }

        prepared_initial_active_features =
            std::move(initial_active_features);

        prepared_allowed_proxy_features =
            std::move(proxy_active_features);

        numerical_X_cols_for_greedy = std::move(numerical_cols_for_greedy);
        numerical_global_sorted_idx = std::move(numerical_global_sorted_idx_local);
        numerical_unique_values_for_greedy =
            std::move(numerical_unique_values_for_greedy_local);

        if (numerical_X_cols_for_greedy.size() != prepared_continuous_starts.size() ||
            numerical_global_sorted_idx.size() != prepared_continuous_starts.size() ||
            numerical_unique_values_for_greedy.size() != prepared_continuous_starts.size()) {
            throw std::logic_error(
                "Numerical greedy arrays do not align with prepared_continuous_starts."
            );
        }

        has_prepared_data = true;
    }

    std::vector<double> fit_repeated_subsamples(
        const std::vector<std::vector<bool>>& X_col_major,
        const std::vector<int>& y,
        double lambda,
        int8_t depth_budget,
        double rashomon_mult,
        int8_t lookahead_k,
        bool use_multipass_flag,
        bool rule_list_mode_flag,
        int proxy_style_in,
        bool majority_leaf_only_flag,
        bool cache_cheap_subproblems_flag,
        bool proxy_caching_flag,
        const std::vector<int>& allowed_proxy_features_in,
        bool restrict_proxy_in_lickety_in,
        bool restrict_proxy_in_depthd_exact_in,
        bool restrict_proxy_in_greedy_in,
        const std::vector<int>& continuous_starts_in,
        double subsample_fraction,
        int num_subsamples,
        uint64_t seed,
        bool reuse_caches_between_subsamples,
        bool stronger_rollout_flag = false,
        bool use_deferral_flag = false,
        double eta_defer_in = 0.0,
        const std::vector<int>& bb_pred = {}
    ) {
        // start both benchmark cases from a completely clean object
        clear_fit_state_();

        if (X_col_major.empty()) {
            throw std::runtime_error("X_col_major is empty.");
        }

        if (X_col_major[0].empty()) {
            throw std::runtime_error("X_col_major has zero samples.");
        }

        if (
            !std::isfinite(subsample_fraction) ||
            subsample_fraction <= 0.0 ||
            subsample_fraction > 1.0
        ) {
            throw std::runtime_error(
                "subsample_fraction must lie in (0, 1]."
            );
        }

        if (num_subsamples <= 0) {
            throw std::runtime_error(
                "num_subsamples must be positive."
            );
        }

        n_features = (int)X_col_major.size();
        n_samples = (int)X_col_major[0].size();

        if ((int)y.size() != n_samples) {
            throw std::runtime_error(
                "y length does not match number of samples."
            );
        }

        for (int f = 1; f < n_features; ++f) {
            if (
                (int)X_col_major[(size_t)f].size() !=
                n_samples
            ) {
                throw std::runtime_error(
                    "X_col_major columns have different "
                    "numbers of samples."
                );
            }
        }

        continuous_starts = continuous_starts_in;
        y_train = y;

        if (
            greedy_continuous_mode ==
            GreedyContinuousMode::NUMERICAL
        ) {
            if (
                numerical_X_cols_for_greedy.size() !=
                    continuous_starts.size() ||
                numerical_global_sorted_idx.size() !=
                    continuous_starts.size() ||
                numerical_unique_values_for_greedy.size() !=
                    continuous_starts.size()
            ) {
                throw std::runtime_error(
                    "Numerical greedy arrays must align "
                    "one-to-one with continuous_starts."
                );
            }
        }

        // full original-data bitvector dimensions.

        n_words = (n_samples + 63) / 64;

        tail_mask =
            (n_samples % 64)
                ? ((1ULL << (n_samples % 64)) - 1ULL)
                : ~0ULL;

        const int subsample_size =
            std::max(
                1,
                std::min(
                    n_samples,
                    (int)std::llround(
                        subsample_fraction *
                        (double)n_samples
                    )
                )
            );

        // gamma is scaled by local subproblem size
        gamma = (int)std::llround(
            lambda * (double)subsample_size
        );

        trained_depth_budget = depth_budget;
        lookahead_init = lookahead_k;
        use_multipass = use_multipass_flag;
        rule_list_mode = rule_list_mode_flag;
        majority_leaf_only = majority_leaf_only_flag;
        cache_cheap_subproblems =
            cache_cheap_subproblems_flag;
        proxy_style = proxy_style_in;
        proxy_caching_enabled = proxy_caching_flag;
        stronger_rollout = stronger_rollout_flag;

        restrict_proxy_in_lickety =
            restrict_proxy_in_lickety_in;

        restrict_proxy_in_depthd_exact =
            restrict_proxy_in_depthd_exact_in;

        restrict_proxy_in_greedy =
            restrict_proxy_in_greedy_in;

        allowed_proxy_features.clear();
        allowed_proxy_features.reserve(
            allowed_proxy_features_in.size()
        );

        for (int f : allowed_proxy_features_in) {
            if (f < 0 || f >= n_features) {
                throw std::runtime_error(
                    "allowed_proxy_features contains "
                    "out-of-range feature index."
                );
            }

            allowed_proxy_features.push_back(f);
        }

        std::sort(
            allowed_proxy_features.begin(),
            allowed_proxy_features.end()
        );

        allowed_proxy_features.erase(
            std::unique(
                allowed_proxy_features.begin(),
                allowed_proxy_features.end()
            ),
            allowed_proxy_features.end()
        );

        use_deferral = use_deferral_flag;
        eta_defer = eta_defer_in;

        if (use_deferral) {
            if ((int)bb_pred.size() != n_samples) {
                throw std::runtime_error(
                    "fit_repeated_subsamples: "
                    "use_deferral=true requires bb_pred "
                    "with the same length as y."
                );
            }

            if (
                !std::isfinite(eta_defer) ||
                eta_defer < 0.0
            ) {
                throw std::runtime_error(
                    "fit_repeated_subsamples: eta_defer "
                    "must be finite and nonnegative."
                );
            }
        }

        X_bits.assign(
            (size_t)n_features,
            Packed((size_t)n_words)
        );

        for (int f = 0; f < n_features; ++f) {
            auto& col = X_bits[(size_t)f].w;

            for (int i = 0; i < n_samples; ++i) {
                if (
                    X_col_major[(size_t)f][(size_t)i]
                ) {
                    col[(size_t)(i >> 6)] |=
                        (1ULL << (i & 63));
                }
            }

            col[(size_t)(n_words - 1)] &=
                tail_mask;
        }

        int y_max = 0;

        for (int i = 0; i < n_samples; ++i) {
            y_max = std::max(
                y_max,
                y[(size_t)i]
            );
        }

        num_classes = y_max + 1;

        Y_bits.assign(
            (size_t)num_classes,
            Packed((size_t)n_words)
        );

        for (int i = 0; i < n_samples; ++i) {
            const int yi = y[(size_t)i];

            if (yi < 0 || yi >= num_classes) {
                throw std::runtime_error(
                    "y contains an invalid class label."
                );
            }

            Y_bits[(size_t)yi]
                .w[(size_t)(i >> 6)] |=
                (1ULL << (i & 63));
        }

        for (int c = 0; c < num_classes; ++c) {
            Y_bits[(size_t)c]
                .w[(size_t)(n_words - 1)] &=
                tail_mask;
        }

        BBwrong = Packed((size_t)n_words);
        BBwrong.clear();

        if (use_deferral) {
            for (int i = 0; i < n_samples; ++i) {
                const int bi =
                    bb_pred[(size_t)i];

                if (
                    bi < 0 ||
                    bi >= num_classes
                ) {
                    throw std::runtime_error(
                        "bb_pred values must lie in "
                        "the same class range as y."
                    );
                }

                if (bi != y[(size_t)i]) {
                    BBwrong
                        .w[(size_t)(i >> 6)] |=
                        (1ULL << (i & 63));
                }
            }

            BBwrong
                .w[(size_t)(n_words - 1)] &=
                tail_mask;
        }

        if (proxy_style == 2) {
            k_at_depth.assign(
                (size_t)depth_budget + 1,
                1
            );

            int K = lookahead_init;
            int kk = K;

            for (
                int d = depth_budget;
                d >= 0;
                --d
            ) {
                k_at_depth[(size_t)d] =
                    std::min(d, kk);

                kk =
                    (kk > 1)
                        ? (kk - 1)
                        : K;
            }

        } else {
            k_at_depth.clear();
        }

        reserve_caches_mid_();

        const PathKey& root_pk =
            empty_pk();

        const ContinuousPath& root_cpath =
            empty_continuous_path();

        std::vector<int> B_active_all =
            all_feature_indices_();

        std::vector<int> row_order(
            (size_t)n_samples
        );

        std::mt19937_64 rng(seed);

        std::vector<double> cumulative_times;
        cumulative_times.reserve(
            (size_t)num_subsamples
        );

        double cumulative_seconds = 0.0;

        for (
            int rep = 0;
            rep < num_subsamples;
            ++rep
        ) {
            // reset to [0, 1, ..., N-1] before every shuffle.
            for (int i = 0; i < n_samples; ++i) {
                row_order[(size_t)i] = i;
            }

            std::shuffle(
                row_order.begin(),
                row_order.end(),
                rng
            );

            // root bitvector is N bits wide but only p*N are 1.
            Packed root((size_t)n_words);

            for (
                int j = 0;
                j < subsample_size;
                ++j
            ) {
                const int row =
                    row_order[(size_t)j];

                root.w[(size_t)(row >> 6)] |=
                    (1ULL << (row & 63));
            }

            root.w[(size_t)(n_words - 1)] &=
                tail_mask;

            const uint64_t subsample_hash =
                hash_mask64(
                    root.w.data(),
                    n_words,
                    tail_mask
                );

            const auto run_start =
                std::chrono::steady_clock::now();


            result.reset();

            // case 1: clear proxy caches etc
            if (
                !reuse_caches_between_subsamples
            ) {
                trie_cache.clear();
                greedy_cache.clear();
                

                lickety_cache_k2.clear();
                lickety_cache_kla.clear();

                greedy_first_split_cache.clear();

                anytime_lickety_first_split_cache.clear();

                continuous_proxy_completion_cache.clear();

                mask_ids = MaskIdTable();
                lit_ids = LitIdTable();

                fingerprint128_ids =
                    Fingerprint128IdTable();
            }

            // case 2: dont clear caches
        
            anytime_mode_active_ = false;

            single_tree_refined_override_.reset();

            best_objective = 0;
            obj_bound = 0;

            if (lookahead_init <= 0) {
                best_objective =
                    greedy_proxy_objective_(
                        root,
                        depth_budget,
                        root_pk,
                        root_cpath
                    );

            } else if (proxy_style == 4) {
                best_objective =
                    split_algorithm(
                        root,
                        depth_budget,
                        lookahead_init,
                        root_pk
                    );

            } else {
                best_objective =
                    lickety_proxy_objective_(
                        root,
                        depth_budget,
                        lookahead_init,
                        root_pk,
                        root_cpath
                    );
            }

           obj_bound = rashomon_budget_(
                best_objective,
                rashomon_mult,
                subsample_size,
                true
            );

            // fit the actual Rashomon set on the subsample mask.

            result =
                construct_trie(
                    root,
                    depth_budget,
                    obj_bound,
                    root_pk,
                    root_cpath,
                    &B_active_all
                );

            const double run_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    run_start
                ).count();

            cumulative_seconds +=
                run_seconds;

            cumulative_times.push_back(
                cumulative_seconds
            );

            const int min_obj =
                result
                    ? result->min_objective
                    : std::numeric_limits<int>::max();

            std::cout
                << "Subsample "
                << (rep + 1)
                << "/"
                << num_subsamples

                << " | mode="
                << (
                    reuse_caches_between_subsamples
                        ? "reuse-all-caches"
                        : "clear-all-caches"
                )

                << " | rows="
                << subsample_size
                << "/"
                << n_samples

                << " | sample_hash="
                << subsample_hash

                << " | best="
                << best_objective

                << " | bound="
                << obj_bound

                << " | min="
                << min_obj

                << " | run_seconds="
                << run_seconds

                << " | cumulative_seconds="
                << cumulative_seconds

                << " | greedy_cache="
                << greedy_cache.size()

                << " | lickety_cache="
                << (
                    use_kla_cache()
                        ? lickety_cache_kla.size()
                        : lickety_cache_k2.size()
                )

                << " | cont_proxy_cache="
                << continuous_proxy_completion_cache.size()

                << " | trie_cache="
                << trie_cache.size()

                << "\n";
        }

        return cumulative_times;
    }

    std::vector<double> fit_prepared_repeated_subsamples(
        double lambda,
        int8_t depth_budget,
        double rashomon_mult,
        int8_t lookahead_k,
        bool use_multipass_flag,
        bool rule_list_mode_flag,
        int proxy_style_in,
        bool majority_leaf_only_flag,
        bool cache_cheap_subproblems_flag,
        bool proxy_caching_flag,
        bool restrict_proxy_in_lickety_in,
        bool restrict_proxy_in_depthd_exact_in,
        bool restrict_proxy_in_greedy_in,
        double subsample_fraction,
        int num_subsamples,
        uint64_t seed,
        bool reuse_caches_between_subsamples,
        bool stronger_rollout_flag = false,
        bool use_deferral_flag = false,
        double eta_defer_in = 0.0
    ) {
        if (!has_prepared_data) {
            throw std::runtime_error(
                "No prepared data. Call prepare_continuous_data(...) "
                "before fit_prepared_repeated_subsamples(...)."
            );
        }

        return fit_repeated_subsamples(
            prepared_X_col_major,
            prepared_y,
            lambda,
            depth_budget,
            rashomon_mult,
            lookahead_k,
            use_multipass_flag,
            rule_list_mode_flag,
            proxy_style_in,
            majority_leaf_only_flag,
            cache_cheap_subproblems_flag,
            proxy_caching_flag,
            prepared_allowed_proxy_features,
            restrict_proxy_in_lickety_in,
            restrict_proxy_in_depthd_exact_in,
            restrict_proxy_in_greedy_in,
            prepared_continuous_starts,
            subsample_fraction,
            num_subsamples,
            seed,
            reuse_caches_between_subsamples,
            stronger_rollout_flag,
            use_deferral_flag,
            eta_defer_in,
            prepared_bb_pred
        );
    }

    // use the already prepared data and fit the rashomon set (by calling fit, which will then do the equivalent of ContinuousRSet)
    void fit_prepared(
        double lambda,
        int8_t depth_budget,
        double rashomon_mult,
        int8_t lookahead_k,
        int root_budget,
        bool use_multipass_flag,
        bool rule_list_mode_flag,
        int proxy_style_in,
        bool majority_leaf_only_flag,
        bool cache_cheap_subproblems_flag,
        bool proxy_caching_flag,
        bool restrict_proxy_in_lickety_in,
        bool restrict_proxy_in_depthd_exact_in,
        bool restrict_proxy_in_greedy_in,
        bool rashomon_mode,
        bool stronger_rollout_flag=false,
        bool use_deferral_flag = false,
        double eta_defer_in = 0.0
    ) {
        if (!has_prepared_data) {
            throw std::runtime_error("No prepared data. Call prepare_continuous_data(...) before fit_prepared(...).");
        }

        fit(
            prepared_X_col_major,
            prepared_y,
            lambda,
            depth_budget,
            rashomon_mult,
            lookahead_k,
            root_budget,
            use_multipass_flag,
            rule_list_mode_flag,
            proxy_style_in,
            majority_leaf_only_flag,
            cache_cheap_subproblems_flag,
            proxy_caching_flag,
            prepared_allowed_proxy_features,
            restrict_proxy_in_lickety_in,
            restrict_proxy_in_depthd_exact_in,
            restrict_proxy_in_greedy_in,
            rashomon_mode,
            prepared_continuous_starts,
            stronger_rollout_flag,
            use_deferral_flag,
            eta_defer_in,
            prepared_bb_pred
        );
    }

    std::vector<int> all_feature_indices_() const {
        std::vector<int> out;
        out.reserve((std::size_t)n_features);

        for (int f = 0; f < n_features; ++f) {
            out.push_back(f);
        }

        return out;
    }

    struct ReachableActions {
        uint64_t class_mask = 0;
        bool can_defer = false;
    };


    // initializes the budget based on the proxy or optimal solution and will call construct_true
    // construct_trie is the equivalent of ContinuousRSet, although we will go to a construct_trie_extend variant when extending or when we have a different active feature set
    // this is code duplication but gets us some small speedups because of overhead
    void fit(const std::vector<std::vector<bool>>& X_col_major,
             const std::vector<int>& y,
             double lambda,
             int8_t depth_budget,
             double rashomon_mult,
             int8_t lookahead_k,
             int root_budget,
             bool use_multipass_flag,
             bool rule_list_mode_flag,
             int proxy_style_in,
             bool majority_leaf_only_flag,
             bool cache_cheap_subproblems_flag,
             bool proxy_caching_flag,
             const std::vector<int>& allowed_proxy_features_in,
             bool restrict_proxy_in_lickety_in,
             bool restrict_proxy_in_depthd_exact_in,
             bool restrict_proxy_in_greedy_in,
             bool rashomon_mode,
             const std::vector<int>& continuous_starts_in,
             bool stronger_rollout_flag=false,
             bool use_deferral_flag = false,
             double eta_defer_in = 0.0,
             const std::vector<int>& bb_pred = {}
            ) {

        clear_fit_state_();

        if (X_col_major.empty()) {
            throw std::runtime_error("X_col_major is empty.");
        }
        if (X_col_major[0].empty()) {
            throw std::runtime_error("X_col_major has zero samples.");
        }

        n_features = (int)X_col_major.size();
        n_samples  = (int)X_col_major[0].size();
        continuous_starts = continuous_starts_in;
        y_train = y;
       
        if (greedy_continuous_mode == GreedyContinuousMode::NUMERICAL) {
            if (numerical_X_cols_for_greedy.size() != continuous_starts.size() ||
                numerical_global_sorted_idx.size() != continuous_starts.size() ||
                numerical_unique_values_for_greedy.size() != continuous_starts.size()) {
                throw std::runtime_error(
                    "Numerical greedy arrays must align one-to-one with continuous_starts."
                );
            }
        }
        n_words = (n_samples + 63) / 64; // 64 -> 1, 65 -> 2
        tail_mask = (n_samples % 64) ? ((1ULL << (n_samples % 64)) - 1ULL) : ~0ULL; // if multiple of 64, all 1s. otherwise, n_samples % 64 1s followed by 0s.
        gamma = (int)llround(lambda * (double)n_samples);

        use_deferral = use_deferral_flag;
        eta_defer = eta_defer_in;

        if (use_deferral) {
            if ((int)bb_pred.size() != n_samples) {
                throw std::runtime_error(
                    "fit: use_deferral=true requires bb_pred with the same "
                    "length as y."
                );
            }

            if (!std::isfinite(eta_defer) || eta_defer < 0.0) {
                throw std::runtime_error(
                    "fit: eta_defer must be finite and nonnegative."
                );
            }
        }

        trained_depth_budget = depth_budget;
        lookahead_init = lookahead_k;
        use_multipass = use_multipass_flag;
        rule_list_mode = rule_list_mode_flag;
        majority_leaf_only = majority_leaf_only_flag;
        cache_cheap_subproblems = cache_cheap_subproblems_flag;
        proxy_style = proxy_style_in;
        proxy_caching_enabled = proxy_caching_flag;
        stronger_rollout = stronger_rollout_flag;
        if (!rashomon_mode) { // force proxy caching on in single-tree mode - required for this codebase
            proxy_caching_enabled = true;
            cache_cheap_subproblems = true;
        } 

        reserve_caches_mid_();

        restrict_proxy_in_lickety = restrict_proxy_in_lickety_in;
        restrict_proxy_in_depthd_exact = restrict_proxy_in_depthd_exact_in;
        restrict_proxy_in_greedy = restrict_proxy_in_greedy_in;

        allowed_proxy_features.clear();
        allowed_proxy_features.reserve(allowed_proxy_features_in.size());

        for (int f : allowed_proxy_features_in) {
            if (f < 0 || f >= n_features) {
                throw std::runtime_error("allowed_proxy_features contains out-of-range feature index.");
            }
            allowed_proxy_features.push_back(f);
        }

        std::sort(allowed_proxy_features.begin(), allowed_proxy_features.end());
        allowed_proxy_features.erase(
            std::unique(allowed_proxy_features.begin(), allowed_proxy_features.end()),
            allowed_proxy_features.end()
        );

        // if (num_proxy_features_in <= 0) num_proxy_features = n_features;
        // else num_proxy_features = std::min(num_proxy_features_in, n_features);

        X_bits.assign(n_features, Packed(n_words)); // length n_features with entries of Packed, initialized to all 0s bits, we will set below.
        for (int f = 0; f < n_features; ++f) {
            auto &col = X_bits[f].w; // reference to the array of 64-bit words for that feature
            for (int i = 0; i < n_samples; ++i) {
                if (X_col_major[f][i]) col[i>>6] |= (1ULL << (i & 63)); // i>>6 integer division by 64 to answer what word are we in. then i & 63 = i % 64 to get index within word. (1ULL << (i & 63)) creates a 64-bit mask with exactly one bit = 1 at the position you need, then do bitwise or with ol[i >> 6] to set the position to 1 if it is not already set.
                // if feature f is true for sample i, set the bit corresponding to row i to true in the packed column by doing bitwise or with the current column and a 64 bit word with exactly one 1.
            }
            col[n_words-1] &= tail_mask; // bitwise and to 0 out invalid
        }

        // so the format is now a contiguous array of 64 bit words. if our remainder is 5 (and we are in word x), then our bit is the 5th least significant one in the word.
        // so increasing row index increases the bit position within a word.
        // last word is padded with 0 (at the front, most significant bits)

        // Ypos = Packed(n_words);
        // for (int i = 0; i < n_samples; ++i) {
        //     if (y[i]) Ypos.w[i>>6] |= (1ULL << (i & 63));
        // }
        // Ypos.w[n_words-1] &= tail_mask;
        int y_max = 0;
        for (int i = 0; i < n_samples; ++i) y_max = std::max(y_max, y[i]);
        num_classes = y_max + 1;

        Y_bits.assign((size_t)num_classes, Packed(n_words));
        for (int c = 0; c < num_classes; ++c) {
            Y_bits[(size_t)c].clear();
        }

        for (int i = 0; i < n_samples; ++i) {
            const int yi = y[i];
            Y_bits[(size_t)yi].w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
        }
        for (int c = 0; c < num_classes; ++c) {
            Y_bits[(size_t)c].w[(size_t)(n_words - 1)] &= tail_mask;
        }

        Packed root(n_words);
        for (int i = 0; i < n_words-1; ++i) root.w[i] = ~0ULL; // not 0, so all 1.
        root.w[n_words-1] = tail_mask; // enforce 0s for out of scope

        BBwrong = Packed((size_t)n_words);
        BBwrong.clear();

        if (use_deferral) {
            for (int i = 0; i < n_samples; ++i) {
                const int bi = bb_pred[(size_t)i];

                if (bi < 0 || bi >= num_classes) {
                    throw std::runtime_error(
                        "fit: bb_pred values must lie in the same class "
                        "range as y."
                    );
                }

                if (bi != y[(size_t)i]) {
                    BBwrong.w[(size_t)(i >> 6)] |=
                        (1ULL << (i & 63));
                }
            }

            BBwrong.w[(size_t)(n_words - 1)] &= tail_mask;
        }

        const PathKey& root_pk = empty_pk();
        const ContinuousPath& root_cpath = empty_continuous_path();

        // SINGLE TREE SUPPORT - always use all features
        if (!rashomon_mode) {
            int single_obj = 0;
            if (lookahead_init <= 0) {
                single_obj = train_greedy_continuous(root, depth_budget, root_pk, root_cpath);
            } else {
                if (proxy_style == 4) {
                    single_obj = split_algorithm(root, depth_budget, lookahead_init, root_pk);
                } else {
                    single_obj = generalized_lickety_split_continuous(root, depth_budget, lookahead_init, root_pk, root_cpath);
                }
            }
            
            std::cout << "Single-tree objective: " << single_obj
                    << " (" << (double)single_obj / (double)n_samples << ")\n";
            return; // done; do NOT build trie or do rashomon things
        }

        if (root_budget >= 0) {
            // user-specified bound: skip reference solution
            obj_bound = root_budget;
            std::cout << "Objective bound (user-set): " << obj_bound << "\n";

        } else {
            if (lookahead_init <= 0) { // set based on greedy even if our proxy is a leaf
                best_objective = greedy_proxy_objective_(root, depth_budget, root_pk, root_cpath);
            } else {
                if (proxy_style == 4) {
                    best_objective = split_algorithm(root, depth_budget, lookahead_init, root_pk);
                } else {
                    best_objective = lickety_proxy_objective_(root, depth_budget, lookahead_init, root_pk, root_cpath);
                }
            }
            cout << "Best objective: " << best_objective
                << " (" << (double)best_objective / (double)n_samples << ")\n";

            obj_bound = rashomon_budget_(
                best_objective,
                rashomon_mult,
                n_samples,
                true
            );


            cout << "Objective bound: " << obj_bound << "\n";
        }

        if (proxy_style == 2) { // we define to be depth_budget+1 size but depth 0 doesn't actually matter
            k_at_depth.assign(depth_budget + 1, 1);
            int K = lookahead_init;
            int kk = K;
            for (int d = depth_budget; d >= 0; --d) {
                k_at_depth[d] = std::min(d, kk);
                kk = (kk > 1) ? (kk - 1) : K; // increment down then wrap
            }
        } else {
            k_at_depth.clear();
        }

        std::vector<int> B_active_all = all_feature_indices_();
        result = construct_trie(root, depth_budget, obj_bound, root_pk, root_cpath, &B_active_all);

        if (!result) {
            cout << "Minimum objective: NONE (empty Rashomon set)\n";
            cout << "Cache sizes - Greedy: " << greedy_cache.size()
                << ", Lickety: "
                << (use_kla_cache() ? lickety_cache_kla.size() : lickety_cache_k2.size())
                << ", Trie: " << trie_cache.size()
                << ", Trie cache: " << (trie_cache_enabled ? "ON" : "OFF")
                << "\n";
            return;
        }

        // cout << "Found " << result->count_trees() << " trees\n"; // we'll let the user compute this query if they want it because it is somewhat expensive
        cout << "Minimum objective: " << result->min_objective << "\n";
        cout << "Cache sizes - Greedy: " << greedy_cache.size()
            << ", Lickety: "
            << (use_kla_cache() ? lickety_cache_kla.size() : lickety_cache_k2.size())
            << ", Trie: " << trie_cache.size()
            << ", Trie cache: " << (trie_cache_enabled ? "ON" : "OFF")
            << "\n";
        // cout << ", Distinct subproblems (greedy U lickety): " << count_distinct_subproblems_union();
        // if (key_mode == KeyMode::EXACT) {
        //     cout << ", Unique masks: " << mask_ids.size();
        // }
        // if (key_mode == KeyMode::LITS_EXACT) {
        //     cout << ", Unique literal subproblems: " << lit_ids.size();
        // }
    }

    // this is how we evaluate extending subgraphs to a larger budget. we first solve it with the budget fit set, but we can also enlarge it.
    // you can set a bigger varepsilon_mult, as we do in the main paper, or try other ways to enlarge it.
    // we test with mode 3 in the main paper.
    shared_ptr<TreeTrieNode> fit_then_extend(
        const std::vector<std::vector<bool>>& X_col_major,
        const std::vector<int>& y,
        double lambda,
        int8_t depth_budget,
        double first_rashomon_mult,
        double second_rashomon_mult,
        double multiplier_step_size,
        int8_t lookahead_k,
        bool use_multipass_flag,
        bool rule_list_mode_flag,
        int proxy_style_in,
        bool majority_leaf_only_flag,
        bool cache_cheap_subproblems_flag,
        bool proxy_caching_flag,
        const std::vector<int>& allowed_proxy_features_in,
        bool restrict_proxy_in_lickety_in,
        bool restrict_proxy_in_depthd_exact_in,
        bool restrict_proxy_in_greedy_in,
        const std::vector<int>& continuous_starts_in,
        bool stronger_rollout_flag = false,
        double runtime_limit_seconds = -1.0,
        double memory_limit_mb = -1.0,
        bool use_deferral_flag = false,
        double eta_defer_in = 0.0,
        const std::vector<int>& bb_pred = {}
    ) {
        if (
            first_rashomon_mult < 0.0 ||
            second_rashomon_mult < 0.0
        ) {
            throw std::runtime_error(
                "Rashomon multipliers must be nonnegative."
            );
        }

        if (
            second_rashomon_mult > first_rashomon_mult &&
            multiplier_step_size <= 0.0
        ) {
            throw std::runtime_error(
                "multiplier_step_size must be positive when the second "
                "Rashomon multiplier is larger than the first."
            );
        }

        begin_resource_tracking_(
            runtime_limit_seconds,
            memory_limit_mb
        );

        // stage 1: ordinary solve using the first multiplier.
        fit(
            X_col_major,
            y,
            lambda,
            depth_budget,
            first_rashomon_mult,
            lookahead_k,
            /*root_budget=*/-1,
            use_multipass_flag,
            rule_list_mode_flag,
            proxy_style_in,
            majority_leaf_only_flag,
            cache_cheap_subproblems_flag,
            proxy_caching_flag,
            allowed_proxy_features_in,
            restrict_proxy_in_lickety_in,
            restrict_proxy_in_depthd_exact_in,
            restrict_proxy_in_greedy_in,
            /*rashomon_mode=*/true,
            continuous_starts_in,
            stronger_rollout_flag,
            use_deferral_flag,
            eta_defer_in,
            bb_pred
        );

        if (!result) {
            throw std::runtime_error(
                "The first-stage fit did not construct a Rashomon graph."
            );
        }

        const int first_budget = obj_bound;
        int current_budget = first_budget;

        Packed root = root_mask_();

        const PathKey& root_pk = empty_pk();
        const ContinuousPath& root_cpath =
            empty_continuous_path();

        std::vector<int> all_features =
            all_feature_indices_();

        constexpr double multiplier_tolerance = 1e-12;

        const bool same_multiplier =
            std::abs(
                second_rashomon_mult -
                first_rashomon_mult
            ) <= multiplier_tolerance;

        const bool smaller_multiplier =
            second_rashomon_mult <
            first_rashomon_mult -
            multiplier_tolerance;

        std::cout
            << "First objective bound: "
            << first_budget
            << "\n";

        // mode 1: equal multipliers.
        // repeatedly inspect split-level proxy/minimum ratios.
        // after every extension, recompute them on the larger graph.
        // stop when the candidate budget is not larger.
        if (same_multiplier) {
            int round = 0;

            while (true) {
                ++round;

                std::cout
                    << "Split-gap extension round "
                    << round
                    << "\n";

                const int candidate_budget =
                    split_gap_extension_budget_(
                        result,
                        root,
                        depth_budget,
                        first_budget,
                        current_budget,
                        root_pk,
                        root_cpath
                    );

                if (candidate_budget <= current_budget) {
                    std::cout
                        << "Split-gap extension converged at budget "
                        << current_budget
                        << "\n";
                    break;
                }

                std::cout
                    << "Extending from objective bound "
                    << current_budget
                    << " to "
                    << candidate_budget
                    << "\n";

                extend_result_to_budget_(
                    depth_budget,
                    candidate_budget,
                    root,
                    root_pk,
                    root_cpath,
                    all_features
                );

                current_budget = candidate_budget;
            }

            obj_bound = current_budget;
            return result;
        }

        // mode 2: second multiplier is smaller.
        // the smaller multiplier is a mode selector. it is not used
        // as a smaller target budget. repeatedly inspect each
        // treetrienode's own proxy/minimum ratio.
        if (smaller_multiplier) {
            int round = 0;

            while (true) {
                ++round;

                std::cout
                    << "Trie-node-gap extension round "
                    << round
                    << "\n";

                const int candidate_budget =
                    trie_node_gap_extension_budget_(
                        result,
                        root,
                        depth_budget,
                        first_budget,
                        current_budget,
                        root_pk,
                        root_cpath
                    );

                if (candidate_budget <= current_budget) {
                    std::cout
                        << "Trie-node-gap extension converged at budget "
                        << current_budget
                        << "\n";
                    break;
                }

                std::cout
                    << "Extending from objective bound "
                    << current_budget
                    << " to "
                    << candidate_budget
                    << "\n";

                extend_result_to_budget_(
                    depth_budget,
                    candidate_budget,
                    root,
                    root_pk,
                    root_cpath,
                    all_features
                );

                current_budget = candidate_budget;
            }

            obj_bound = current_budget;
            return result;
        }

        // mode 3: second multiplier is larger.
        // do expected behavior of extending to it
        const int explicit_second_budget =
            rashomon_budget_(
                best_objective,
                second_rashomon_mult,
                n_samples,
                true
            );

        if (explicit_second_budget < first_budget) {
            throw std::logic_error(
                "The explicit second objective bound is unexpectedly "
                "smaller than the first objective bound."
            );
        }

        if (explicit_second_budget == first_budget) {
            obj_bound = first_budget;
            end_resource_tracking_();
            return result;
        }

        double current_multiplier = first_rashomon_mult;

        while (current_budget < explicit_second_budget) {
            if (resource_limit_reached_()) {
                std::cout
                    << "Resource limit reached; stopping extension.\n";
                break;
            }

            current_multiplier = std::min(
                second_rashomon_mult,
                current_multiplier + multiplier_step_size
            );

            int next_budget = rashomon_budget_(
                best_objective,
                current_multiplier,
                n_samples,
                true
            );

            // ensure every unfinished iteration makes integer-budget progress.
            if (
                next_budget <= current_budget
            ) {
                next_budget = current_budget + 1;
            }

            // never exceed the requested final budget.
            next_budget = std::min(
                next_budget,
                explicit_second_budget
            );

            std::cout
                << "Extending multiplier to "
                << current_multiplier
                << " with objective bound "
                << next_budget
                << "\n";

            extend_result_to_budget_(
                depth_budget,
                next_budget,
                root,
                root_pk,
                root_cpath,
                all_features
            );

            if (resource_limit_reached_()) {
                std::cout
                    << "Resource limit reached during extension; "
                    << "keeping completed objective bound "
                    << current_budget
                    << ".\n";
                break;
            }


            current_budget = next_budget;
        }
        

        obj_bound = current_budget;
        end_resource_tracking_();

        return result;
    }

    shared_ptr<TreeTrieNode> fit_prepared_then_extend(
        double lambda,
        int8_t depth_budget,
        double first_rashomon_mult,
        double second_rashomon_mult,
        double multiplier_step_size,
        int8_t lookahead_k,
        bool use_multipass_flag,
        bool rule_list_mode_flag,
        int proxy_style_in,
        bool majority_leaf_only_flag,
        bool cache_cheap_subproblems_flag,
        bool proxy_caching_flag,
        bool restrict_proxy_in_lickety_in,
        bool restrict_proxy_in_depthd_exact_in,
        bool restrict_proxy_in_greedy_in,
        bool stronger_rollout_flag = false,
        double runtime_limit_seconds = -1.0,
        double memory_limit_mb = -1.0,
        bool use_deferral_flag = false,
        double eta_defer_in = 0.0
    ) {
        if (!has_prepared_data) {
            throw std::runtime_error(
                "No prepared data. Call prepare_continuous_data(...) "
                "before fit_prepared_then_extend(...)."
            );
        }

        return fit_then_extend(
            prepared_X_col_major,
            prepared_y,
            lambda,
            depth_budget,
            first_rashomon_mult,
            second_rashomon_mult,
            multiplier_step_size,
            lookahead_k,
            use_multipass_flag,
            rule_list_mode_flag,
            proxy_style_in,
            majority_leaf_only_flag,
            cache_cheap_subproblems_flag,
            proxy_caching_flag,
            prepared_allowed_proxy_features,
            restrict_proxy_in_lickety_in,
            restrict_proxy_in_depthd_exact_in,
            restrict_proxy_in_greedy_in,
            prepared_continuous_starts,
            stronger_rollout_flag,
            runtime_limit_seconds,
            memory_limit_mb,
            use_deferral_flag,
            eta_defer_in,
            prepared_bb_pred
        );
    }

    void fit_anytime(
        const std::vector<std::vector<bool>>& X_col_major,
        const std::vector<int>& y,
        double lambda,
        int8_t depth_budget,
        double rashomon_mult,
        double second_rashomon_mult,
        double multiplier_step_size,
        int8_t lookahead_k,
        bool use_multipass_flag,
        bool rule_list_mode_flag,
        int proxy_style_in,
        bool majority_leaf_only_flag,
        bool cache_cheap_subproblems_flag,
        bool proxy_caching_flag,
        const std::vector<int>& proxy_threshold_features,
        const std::vector<int>& initial_active_threshold_features,
        int refinement_width,
        int max_refinement_rounds,
        int proxy_refinement_mode,
        bool continuous_proxy_in_lickety,
        bool continuous_proxy_in_depthd_exact,
        bool continuous_proxy_in_greedy,
        const std::vector<int>& continuous_starts_in,
        double runtime_limit_seconds = -1.0,
        double memory_limit_mb = -1.0,
        bool use_deferral_flag = false,
        double eta_defer_in = 0.0,
        const std::vector<int>& bb_pred = {}
    ) {
        clear_fit_state_();

        if (X_col_major.empty()) {
            throw std::runtime_error("X_col_major is empty.");
        }

        if (X_col_major[0].empty()) {
            throw std::runtime_error("X_col_major has zero samples.");
        }

        n_features = (int)X_col_major.size();
        n_samples  = (int)X_col_major[0].size();
        continuous_starts = continuous_starts_in;
        y_train = y;

        if ((int)y.size() != n_samples) {
            throw std::runtime_error("y length does not match number of samples.");
        }

        for (int f = 1; f < n_features; ++f) {
            if ((int)X_col_major[(std::size_t)f].size() != n_samples) {
                throw std::runtime_error("X_col_major must be rectangular.");
            }
        }

        if (greedy_continuous_mode == GreedyContinuousMode::NUMERICAL) {
            if (numerical_X_cols_for_greedy.size() != continuous_starts.size() ||
                numerical_global_sorted_idx.size() != continuous_starts.size() ||
                numerical_unique_values_for_greedy.size() != continuous_starts.size()) {
                throw std::runtime_error(
                    "Numerical greedy arrays must align one-to-one with continuous_starts."
                );
            }
        }

        if (
            proxy_refinement_mode < 0 ||
            proxy_refinement_mode > 2
        ) {
            throw std::runtime_error(
                "proxy_refinement_mode must be 0, 1, or 2."
            );
        }

        if (
            second_rashomon_mult > rashomon_mult &&
            multiplier_step_size <= 0.0
        ) {
            throw std::runtime_error(
                "multiplier_step_size must be positive when "
                "second_rashomon_mult exceeds rashomon_mult."
            );
        }


        begin_resource_tracking_(
            runtime_limit_seconds,
            memory_limit_mb
        );


        n_words = (n_samples + 63) / 64;
        tail_mask = (n_samples % 64)
            ? ((1ULL << (n_samples % 64)) - 1ULL)
            : ~0ULL;

        gamma = (int)llround(lambda * (double)n_samples);

        use_deferral = use_deferral_flag;
        eta_defer = eta_defer_in;

        if (use_deferral) {
            if ((int)bb_pred.size() != n_samples) {
                throw std::runtime_error(
                    "fit_anytime: use_deferral=true requires "
                    "bb_pred with the same length as y."
                );
            }

            if (
                !std::isfinite(eta_defer) ||
                eta_defer < 0.0
            ) {
                throw std::runtime_error(
                    "fit_anytime: eta_defer must be finite "
                    "and nonnegative."
                );
            }
        }


        trained_depth_budget = depth_budget;
        lookahead_init = lookahead_k;
        use_multipass = use_multipass_flag;
        rule_list_mode = rule_list_mode_flag;
        majority_leaf_only = majority_leaf_only_flag;
        cache_cheap_subproblems = cache_cheap_subproblems_flag;
        proxy_style = proxy_style_in;
        proxy_caching_enabled = proxy_caching_flag;

        if (!proxy_caching_enabled) { // we need this
            proxy_caching_enabled = true;
        }

        reserve_caches_mid_();

        // anytime_continuous_rset will update this
        restrict_proxy_in_lickety = false;
        restrict_proxy_in_depthd_exact = false;
        restrict_proxy_in_greedy = false;
        allowed_proxy_features.clear();

        X_bits.assign(n_features, Packed(n_words));

        for (int f = 0; f < n_features; ++f) {
            auto& col = X_bits[(std::size_t)f].w;

            for (int i = 0; i < n_samples; ++i) {
                if (X_col_major[(std::size_t)f][(std::size_t)i]) {
                    col[(std::size_t)(i >> 6)] |= (1ULL << (i & 63));
                }
            }

            col[(std::size_t)(n_words - 1)] &= tail_mask;
        }

        int y_max = 0;
        for (int i = 0; i < n_samples; ++i) {
            if (y[(std::size_t)i] < 0) {
                throw std::runtime_error("y contains negative class labels.");
            }

            y_max = std::max(y_max, y[(std::size_t)i]);
        }

        num_classes = y_max + 1;

        Y_bits.assign((std::size_t)num_classes, Packed(n_words));

        for (int c = 0; c < num_classes; ++c) {
            Y_bits[(std::size_t)c].clear();
        }

        for (int i = 0; i < n_samples; ++i) {
            const int yi = y[(std::size_t)i];
            Y_bits[(std::size_t)yi].w[(std::size_t)(i >> 6)] |=
                (1ULL << (i & 63));
        }

        for (int c = 0; c < num_classes; ++c) {
            Y_bits[(std::size_t)c].w[(std::size_t)(n_words - 1)] &= tail_mask;
        }

        if (proxy_style == 2) {
            k_at_depth.assign((std::size_t)depth_budget + 1, 1);

            int K = lookahead_init;
            int kk = K;

            for (int d = depth_budget; d >= 0; --d) {
                k_at_depth[(std::size_t)d] = std::min(d, kk);
                kk = (kk > 1) ? (kk - 1) : K;
            }
        } else {
            k_at_depth.clear();
        }

        BBwrong = Packed((size_t)n_words);
        BBwrong.clear();

        if (use_deferral) {
            for (int i = 0; i < n_samples; ++i) {
                const int bi =
                    bb_pred[(size_t)i];

                if (
                    bi < 0 ||
                    bi >= num_classes
                ) {
                    throw std::runtime_error(
                        "fit_anytime: bb_pred values must "
                        "lie in the same class range as y."
                    );
                }

                if (bi != y[(size_t)i]) {
                    BBwrong.w[(size_t)(i >> 6)] |=
                        (1ULL << (i & 63));
                }
            }

            BBwrong.w[
                (size_t)(n_words - 1)
            ] &= tail_mask;
        }

        result = anytime_continuous_rset(
            depth_budget,
            rashomon_mult,
            second_rashomon_mult,
            multiplier_step_size,
            proxy_threshold_features,
            initial_active_threshold_features,
            refinement_width,
            max_refinement_rounds,
            proxy_refinement_mode,
            continuous_proxy_in_lickety,
            continuous_proxy_in_depthd_exact,
            continuous_proxy_in_greedy
        );
        
        const bool stopped_for_resources = resource_limit_reached_();

        end_resource_tracking_();

        if (!result) {
            if (stopped_for_resources) {
                throw std::runtime_error(
                    "Resource limit reached before a feasible graph was constructed."
                );
            }

            throw std::runtime_error(
                "Graph construction returned null without reaching a resource limit."
            );
        }

        cout << "Anytime minimum objective: " << result->min_objective << "\n";
        cout << "Cache sizes - Greedy: " << greedy_cache.size()
            << ", Lickety: "
            << (use_kla_cache() ? lickety_cache_kla.size() : lickety_cache_k2.size())
            << ", Trie: " << trie_cache.size();

        if (key_mode == KeyMode::HASH128) {
            cout << ", Fingerprint128 IDs: " << fingerprint128_ids.size();
        }

        cout << ", Trie cache: " << (trie_cache_enabled ? "ON" : "OFF")
            << "\n";
    }

    // anytime algorithm
    void fit_prepared_anytime(
        double lambda,
        int8_t depth_budget,
        double rashomon_mult,
        double second_rashomon_mult,
        double multiplier_step_size,
        int8_t lookahead_k,
        bool use_multipass_flag,
        bool rule_list_mode_flag,
        int proxy_style_in,
        bool majority_leaf_only_flag,
        bool cache_cheap_subproblems_flag,
        bool proxy_caching_flag,
        const std::vector<int>& proxy_threshold_features,
        const std::vector<int>& initial_active_threshold_features,
        int refinement_width,
        int max_refinement_rounds = -1,
        int proxy_refinement_mode = 0,
        bool continuous_proxy_in_lickety = false,
        bool continuous_proxy_in_depthd_exact = false,
        bool continuous_proxy_in_greedy = false,
        double runtime_limit_seconds = -1.0,
        double memory_limit_mb = -1.0,
        bool use_deferral_flag = false,
        double eta_defer_in = 0.0
    ) {
        if (!has_prepared_data) {
            throw std::runtime_error(
                "No prepared data. Call prepare_continuous_data(...) before fit_prepared_anytime(...)."
            );
        }

        std::vector<int> proxy_feats =
            proxy_threshold_features.empty()
                ? prepared_allowed_proxy_features
                : proxy_threshold_features;

        std::vector<int> initial_feats =
            initial_active_threshold_features.empty()
                ? prepared_initial_active_features
                : initial_active_threshold_features;

  

        fit_anytime(
            prepared_X_col_major,
            prepared_y,
            lambda,
            depth_budget,
            rashomon_mult,
            second_rashomon_mult,
            multiplier_step_size,
            lookahead_k,
            use_multipass_flag,
            rule_list_mode_flag,
            proxy_style_in,
            majority_leaf_only_flag,
            cache_cheap_subproblems_flag,
            proxy_caching_flag,
            proxy_feats,
            initial_feats,
            refinement_width,
            max_refinement_rounds,
            proxy_refinement_mode,
            continuous_proxy_in_lickety,
            continuous_proxy_in_depthd_exact,
            continuous_proxy_in_greedy,
            prepared_continuous_starts,
            runtime_limit_seconds,
            memory_limit_mb,
            use_deferral_flag,
            eta_defer_in,
            prepared_bb_pred
        );
    }

    // the majority of the following extraction methods are existing. we note this and skip to our contributions.

    // predict using the i-th tree in the Rashomon set: X_row_major: binary [n_samples][n_features]
    std::vector<uint8_t> get_predictions(uint64_t tree_index, const std::vector<std::vector<uint8_t>>& X_row_major, const std::vector<int>& bb_pred_row = {}, int defer_placeholder = 99) const {
        const std::size_t n_samples = X_row_major.size();
        if (n_samples == 0) return {};

        const std::size_t n_features = X_row_major[0].size();
        const int8_t depth_budget = trained_depth_budget;

        if ((int)n_features != this->n_features) {
            throw std::runtime_error("Prediction X has different number of features than training.");
        }

        const bool use_placeholder =
            bb_pred_row.empty();

        if (
            !use_placeholder &&
            bb_pred_row.size() != n_samples
        ) {
            throw std::runtime_error(
                "Prediction bb_pred_row has a different "
                "number of rows than X."
            );
        }

        std::shared_ptr<PredNode> tree;

        // single tree mode
        if (!result) {
            if (tree_index != 0) {
                throw std::runtime_error("Single-tree mode only supports tree_index == 0.");
            }
            if (depth_budget < 0) {
                throw std::runtime_error("trained_depth_budget not set. Call fit() first.");
            }

            Packed root((size_t)n_words);
            for (int i = 0; i < n_words - 1; ++i) root.w[(size_t)i] = ~0ULL;
            root.w[(size_t)(n_words - 1)] = tail_mask;

            const PathKey& root_pk = empty_pk();
            tree = single_tree_refined_override_
                ? single_tree_refined_override_
                : build_best_tree_from_caches(root, depth_budget, root_pk);
        } 
        // standard rashomon mode
        else {
            tree = get_ith_tree(tree_index);
        }

        std::vector<uint8_t> out(n_samples, 0);

        std::vector<int> idx(n_samples);
        for (std::size_t i = 0; i < n_samples; ++i) {
            idx[i] = static_cast<int>(i);
        }

        predict_tree_recursive(tree.get(), X_row_major, bb_pred_row, out, idx, use_placeholder, defer_placeholder);
        return out;
    }


    // get predictions from all trees in the rashomon set, as a vector of prediction vectors (one per tree).
    std::vector<std::vector<uint8_t>> get_all_predictions(
        const std::vector<std::vector<uint8_t>>& X_row_major, const std::vector<int>& bb_pred_row = {}, int defer_placeholder = 99
    ) const {
        const uint64_t total =
            result
                ? result->count_trees()
                : 1ULL;
        std::vector<std::vector<uint8_t>> all;
        all.reserve(static_cast<std::size_t>(total));
        for (uint64_t i = 0; i < total; ++i) {
            all.push_back(get_predictions(i, X_row_major, bb_pred_row, defer_placeholder));
        }
        return all;
    }

    std::pair<std::vector<std::vector<int>>, std::vector<int>>
    get_tree_paths(std::uint64_t tree_index) const {
        // single tree mode
        if (!result) {
            if (tree_index != 0) {
                throw std::out_of_range("Single-tree mode only supports tree_index == 0.");
            }

            // root mask - all samples active
            Packed root(n_words);
            for (int i = 0; i < n_words - 1; ++i) root.w[i] = ~0ULL;
            root.w[n_words - 1] = tail_mask;

            const PathKey& root_pk = empty_pk();
            const int8_t depth_budget = trained_depth_budget;

            auto tree = single_tree_refined_override_
                ? single_tree_refined_override_
                : build_best_tree_from_caches(root, depth_budget, root_pk);

            std::vector<std::vector<int>> paths;
            std::vector<int> preds;
            std::vector<int> current;
            collect_paths(tree.get(), current, paths, preds);
            return {paths, preds};
        }

        // standard rashomon mode
        auto tree = get_ith_tree(tree_index);
        std::vector<std::vector<int>> paths;
        std::vector<int> preds;
        std::vector<int> current;

        collect_paths(tree.get(), current, paths, preds);
        return {paths, preds};
    }


    // tAO-style alternating optimization for the reconstructed single tree.
    // each sweep processes internal nodes bottom-up. For a node, descendants are
    // held fixed. A training sample reaching the node receives a temporary binary
    // label only when exactly one child subtree classifies it correctly:
    //   1 -> the current left subtree is correct and the right subtree is wrong
    //   0 -> the current right subtree is correct and the left subtree is wrong
    // samples for which both children are correct or both are wrong are omitted.
    // the temporary classification problem is solved exactly at depth 1 by
    // scanning every existing binary/continuous-threshold column with packed
    // bitvectors. Both polarities are considered. if the reversed polarity wins,
    // the children are swapped so X_bits[f]==1 still follows node->left.
 
    // returns the number of strict node improvements accepted across all sweeps.
    int alternating_optimization(int max_iterations = 10) {
        if (result) {
            throw std::runtime_error(
                "alternating_optimization is only available in single-tree mode."
            );
        }
        if (trained_depth_budget < 0 || n_samples <= 0) {
            throw std::runtime_error("Call fit() before alternating_optimization().");
        }
        if (max_iterations < 0) {
            throw std::invalid_argument("max_iterations must be nonnegative.");
        }
        if (use_deferral) {
            throw std::runtime_error(
                "alternating_optimization currently optimizes classification error only "
                "and is disabled when deferral is active."
            );
        }

        Packed root((size_t)n_words);
        for (int i = 0; i < n_words - 1; ++i) root.w[(size_t)i] = ~0ULL;
        root.w[(size_t)(n_words - 1)] = tail_mask;

        // start from the strongest tree currently represented by the caches unless this method has already been called, in which case continue refining the
        // previous result.
        if (!single_tree_refined_override_) {
            const PathKey& root_pk = empty_pk();
            single_tree_refined_override_ =
                build_best_tree_from_caches(root, trained_depth_budget, root_pk);
        }

        int total_improvements = 0;

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            std::vector<TaoNodeWork_> work;
            work.reserve(64);
            collect_tao_nodes_with_masks_(
                single_tree_refined_override_.get(),
                root,
                0,
                work
            );

            std::stable_sort(
                work.begin(),
                work.end(),
                [](const TaoNodeWork_& a, const TaoNodeWork_& b) {
                    return a.depth > b.depth;
                }
            );

            int improvements_this_sweep = 0;
            for (auto& item : work) {
                if (optimize_tao_node_(item.node, item.mask)) {
                    ++improvements_this_sweep;
                    ++total_improvements;
                }
            }

            if (improvements_this_sweep == 0) break;
        }

        return total_improvements;
    }


    // for individual decision tree algorithm
    std::pair<std::vector<std::vector<int>>, std::vector<int>>
        get_tree_paths_from_tree(const std::shared_ptr<PredNode>& tree) const {
            if (!tree) {
                throw std::runtime_error("Null tree passed to get_tree_paths_from_tree.");
            }

            std::vector<std::vector<int>> paths;
            std::vector<int> preds;
            std::vector<int> current;

            collect_paths(tree.get(), current, paths, preds);
            return {paths, preds};
        }


    // return (unnormalized_objective, normalized_objective) for the ith tree
    std::pair<int, double> get_ith_tree_objective(std::uint64_t i) const {
        // if (!result) {
        //     throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        // }
        if (!result) {
            if (i != 0) throw std::out_of_range("Single-tree mode only supports i==0.");

            if (single_tree_refined_override_) {
                const int refined = tree_training_objective_raw_(single_tree_refined_override_.get());
                return {refined, (double)refined / (double)n_samples};
            }

            if (!proxy_caching_enabled) {
                throw std::runtime_error("Single-tree objective requires proxy_caching_enabled, or recompute objective.");
            }
            if (trained_depth_budget < 0) {
                throw std::runtime_error("trained_depth_budget not set. Call fit() first.");
            }

            // root mask
            Packed root((size_t)n_words);
            for (int w = 0; w < n_words - 1; ++w) root.w[(size_t)w] = ~0ULL;
            root.w[(size_t)(n_words - 1)] = tail_mask;

            const PathKey& root_pk = empty_pk();
            const int8_t d = trained_depth_budget;

            const uint64_t km = key_of_subproblem(root, root_pk);

            int best = std::numeric_limits<int>::max(); // it suffices to return the minimum among the root caches

            // greedy
            if (auto itg = greedy_cache.find(K2{km, d}); itg != greedy_cache.end())
                best = std::min(best, itg->second);

            // lickety
            if (use_kla_cache()) {
                for (int kk = 0; kk <= (int)(d - 1); ++kk) {
                    auto it = lickety_cache_kla.find(KLA{km, d, kk});
                    if (it != lickety_cache_kla.end()) best = std::min(best, it->second);
                }
            } else {
                if (auto it = lickety_cache_k2.find(K2{km, d}); it != lickety_cache_k2.end())
                    best = std::min(best, it->second);
            }

            if (best == std::numeric_limits<int>::max()) {
                throw std::runtime_error("Root objective not found in caches (greedy/lickety).");
            }

            double normalized = (double)best / (double)n_samples;
            return {best, normalized};
        }

        // count_trees will ensure that the histograms are built at the root and every child node (by building them if they are not yet built)
        std::uint64_t total = result->count_trees();
        if (i >= total) {
            throw std::out_of_range("Tree index out of range in get_ith_tree_objective");
        }

        std::uint64_t cum = 0;
        int target_obj = -1;

        // hist is sorted by objective ascending
        for (const auto& e : result->hist) {
            if (i < cum + e.cnt) {
                target_obj = e.obj;
                break;
            }
            cum += e.cnt;
        }

        if (target_obj < 0) {
            throw std::runtime_error("Failed to locate objective bucket in get_ith_tree_objective");
        }

        double normalized = (n_samples > 0)
            ? static_cast<double>(target_obj) / static_cast<double>(n_samples)
            : 0.0;

        return {target_obj, normalized};
    }

    // root LicketySPLIT objective with lookahead=1 so the user can compare frontier cuts to the reference solution
    int root_lickety_objective_lookahead1(int depth_budget) {
        if (n_samples == 0) {
            throw std::runtime_error("Model not fitted.");
        }

        Packed root(n_words);
        for (int i = 0; i < n_words - 1; ++i) root.w[i] = ~0ULL;
        root.w[n_words - 1] = tail_mask;

        PathKey root_pk;
        return generalized_lickety_split_continuous(root, depth_budget, /*k=*/1, root_pk);
    }

    // our anytime rashomon set algorithm which corresponds to algorithm 4 in the main paper.
    // it is a pretty exact translation.
    shared_ptr<TreeTrieNode> anytime_continuous_rset(
        int8_t depth_budget,
        double rashomon_mult,
        double second_rashomon_mult,
        double multiplier_step_size,
        const std::vector<int>& proxy_threshold_features,
        const std::vector<int>& initial_active_threshold_features,
        int refinement_width,
        int max_refinement_rounds = -1,
        int proxy_refinement_mode = 0,
        bool continuous_proxy_in_lickety = false,
        bool continuous_proxy_in_depthd_exact = false,
        bool continuous_proxy_in_greedy = false
    ) {
        if (n_samples <= 0 || n_features <= 0 || n_words <= 0) {
            throw std::runtime_error(
                "anytime_continuous_rset requires prepared/fitted internal data."
            );
        }

        if (depth_budget < 0) {
            throw std::runtime_error("depth_budget must be nonnegative.");
        }

        if (
            proxy_refinement_mode < 0 ||
            proxy_refinement_mode > 2
        ) {
            throw std::runtime_error(
                "proxy_refinement_mode must be 0, 1, or 2."
            );
        }

        if (
            second_rashomon_mult > rashomon_mult &&
            multiplier_step_size <= 0.0
        ) {
            throw std::runtime_error(
                "multiplier_step_size must be positive when extending "
                "to a larger second multiplier."
            );
        }

                const int first_cont = first_continuous_feature_();

        // B_bin = ordinary non-continuous binary features.
        std::vector<int> B_bin;
        B_bin.reserve((std::size_t)first_cont);

        for (int f = 0; f < first_cont; ++f) {
            B_bin.push_back(f);
        }

        std::vector<int> B_proxy =
            proxy_threshold_features;

        std::vector<int> B_initial =
            initial_active_threshold_features;

        for (int f : B_proxy) {
            if (f < 0 || f >= n_features) {
                throw std::runtime_error(
                    "proxy_threshold_features contains "
                    "an out-of-range feature index."
                );
            }
        }

        for (int f : B_initial) {
            if (f < 0 || f >= n_features) {
                throw std::runtime_error(
                    "initial_active_threshold_features contains "
                    "an out-of-range feature index."
                );
            }
        }

        sort_unique_ints_inplace_(B_proxy);
        sort_unique_ints_inplace_(B_initial);

        // B_active = B_bin union B_proxy.
        std::vector<int> B_active;
        B_active.reserve(B_bin.size() + B_initial.size());

        B_active.insert(B_active.end(), B_bin.begin(), B_bin.end());
        B_active.insert(B_active.end(), B_initial.begin(), B_initial.end());

        sort_unique_ints_inplace_(B_active);

        Packed root(n_words);

        for (int w = 0; w < n_words - 1; ++w) {
            root.w[(std::size_t)w] = ~0ULL;
        }

        root.w[(std::size_t)(n_words - 1)] = tail_mask;

        const PathKey& root_pk = empty_pk();
        const ContinuousPath& root_cpath = empty_continuous_path();

        const std::vector<int> old_allowed_proxy_features = allowed_proxy_features;
        const bool old_restrict_lickety = restrict_proxy_in_lickety;
        const bool old_restrict_depthd = restrict_proxy_in_depthd_exact;
        const bool old_restrict_greedy = restrict_proxy_in_greedy;

        allowed_proxy_features = B_proxy;

        restrict_proxy_in_lickety = !continuous_proxy_in_lickety;
        restrict_proxy_in_depthd_exact = !continuous_proxy_in_depthd_exact;
        restrict_proxy_in_greedy = !continuous_proxy_in_greedy;

        anytime_mode_active_ = true;
        anytime_lickety_first_split_cache.clear();

        int P_root;

        if (lookahead_init < 0) {
            P_root = leaf_objective(root);
        } else if (lookahead_init == 0) {
            P_root = greedy_proxy_objective_(
                root,
                depth_budget,
                root_pk,
                root_cpath
            );
        } else {
            P_root = lickety_proxy_objective_(
                root,
                depth_budget,
                lookahead_init,
                root_pk,
                root_cpath
            );
        }

        const int eps_abs = rashomon_budget_(
            P_root,
            rashomon_mult,
            n_samples,
            false
        );


        // don't allow early stopping in the initial solve
        const bool limits_were_active = resource_limits_active_;
        resource_limits_active_ = false;

        shared_ptr<TreeTrieNode> G_min = construct_trie(
            root,
            depth_budget,
            eps_abs,
            root_pk,
            root_cpath,
            &B_active
        );

        resource_limits_active_ = limits_were_active;

        


        int refinement_round = 0;
        while (!active_contains_all_threshold_columns_(B_active)) {
            if (max_refinement_rounds >= 0 &&
                refinement_round >= max_refinement_rounds) {
                break;
            }

            if (resource_limit_reached_()) {
                std::cout
                    << "Resource limit reached; "
                    << "stopping threshold refinement.\n";
                break;
            }

            std::vector<int> B_new = SelectNewThresholds(
                B_active,
                continuous_starts,
                refinement_width
            );

            if (B_new.empty()) {
                break;
            }

            B_active.insert(B_active.end(), B_new.begin(), B_new.end());
            sort_unique_ints_inplace_(B_active);

            // the meaning of complete for this active feature set changed.
            // do not trust old (subproblem, depth) trie-cache hits.
            trie_cache.clear();

            // one shared map for this entire DAG traversal.
            // it is discarded after this refinement round.
            if (!G_min) {
                // RefineGraphDfs doesn't handle nulls, we have to rerun the graph algorithm with more features
                G_min = construct_trie(
                    root,
                    depth_budget,
                    eps_abs,
                    root_pk,
                    root_cpath,
                    &B_active
                );
            } else {
                RefineVisited visited;
                visited.reserve(1024);

                RefineGraphDfs(
                    G_min,
                    root,
                    depth_budget,
                    root_pk,
                    root_cpath,
                    B_active,
                    visited
                );
            }

            ++refinement_round;
        }

        // done using continuous Lickety's k=1 first-split suggestions to augment
        // the active threshold set. from here on, the active set is fixed and
        // lookahead may increase, so the first-split cache should be disabled.
        anytime_mode_active_ = false;
        anytime_lickety_first_split_cache.clear();

        // after all active-threshold refinement is done, we could consider increasing epsilon
        int current_budget = eps_abs;

        if (
            second_rashomon_mult > rashomon_mult &&
            !resource_limit_reached_()
        ) {
            const int final_budget = rashomon_budget_(
                P_root,
                second_rashomon_mult,
                n_samples,
                false
            );

            double current_multiplier = rashomon_mult;

            while (
                current_budget < final_budget &&
                !resource_limit_reached_()
            ) {
                current_multiplier = std::min(
                    second_rashomon_mult,
                    current_multiplier + multiplier_step_size
                );

                int next_budget = rashomon_budget_(
                    P_root,
                    current_multiplier,
                    n_samples,
                    false
                );

                // rounding can otherwise make a multiplier step produce
                // the same integer budget.
                if (next_budget <= current_budget) {
                    next_budget = current_budget + 1;
                }

                next_budget = std::min(
                    next_budget,
                    final_budget
                );

                std::cout
                    << "Extending anytime multiplier to "
                    << current_multiplier
                    << " with objective bound "
                    << next_budget
                    << "\n";

                // extend_result_to_budget_ operates on result, so make sure it
                // points to the current anytime root.
                result = G_min;

                extend_result_to_budget_(
                    depth_budget,
                    next_budget,
                    root,
                    root_pk,
                    root_cpath,
                    B_active
                );

                G_min = result;
                current_budget = next_budget;
            }

            if (resource_limit_reached_()) {
                std::cout
                    << "Resource limit reached; "
                    << "stopping multiplier extension.\n";
            }
        }


        // after all active-threshold refinement is done, optionally increase the
        // proxy lookahead. This can recover splits that were previously pruned by
        // a weaker proxy. The root graph G_min is kept and extended in place.
        // in auto mode, we will decide whether to do this or not

        bool proxy_refinement_stopped_early = false;
        if (proxy_refinement_mode != 0) {

            const int old_lookahead_init = lookahead_init;
            const int final_refinement_lookahead = std::max<int>(lookahead_init, depth_budget - 2); // d-1 is optimal at depth d, but we never evaluate at the root again, always one split lower

            int automatic_refinement_round = 0;

            for (int next_k = lookahead_init + 1;
                next_k <= final_refinement_lookahead;
                ++next_k) {

                if (resource_limit_reached_()) {
                    std::cout
                        << "Resource limit reached; "
                        << "stopping proxy refinement.\n";
                    proxy_refinement_stopped_early = true;
                    break;
                }

                if (proxy_refinement_mode == 2) {
                    const double cumulative_seconds =
                        elapsed_resource_seconds_();

                    const double current_memory_mb =
                        current_memory_mb_();

                    const double scale =
                        std::ldexp(
                            1.0,
                            automatic_refinement_round
                        ); // 2^r

                    const bool time_ok =
                        runtime_limit_seconds_ < 0.0 ||
                        cumulative_seconds /
                            runtime_limit_seconds_
                            <= scale / 500.0;

                    const bool memory_ok =
                        memory_limit_mb_ < 0.0 ||
                        current_memory_mb /
                            memory_limit_mb_
                            <= scale / 37.0;

                    if (!time_ok || !memory_ok) {
                        std::cout
                            << "Further proxy refinement deemed too expensive "
                            << "for automatic mode.\n";

                        proxy_refinement_stopped_early = true;
                        break;
                    }
                }

                lookahead_init = (int8_t)next_k;

                if (proxy_style == 2) {
                    k_at_depth.assign((std::size_t)depth_budget + 1, 1);

                    int K = lookahead_init;
                    int kk = K;

                    for (int dd = depth_budget; dd >= 0; --dd) {
                        k_at_depth[(std::size_t)dd] = std::min(dd, kk);
                        kk = (kk > 1) ? (kk - 1) : K;
                    }
                } else {
                    k_at_depth.clear();
                }

                // E cache: proxy completions are no longer valid because the proxy
                // itself changed when lookahead changed.
                continuous_proxy_completion_cache.clear();

                // subgraph cache: canonical completeness was with respect to the old search
                trie_cache.clear();

                RefineVisited visited;
                visited.reserve(1024);

                RefineGraphDfs(
                    G_min,
                    root,
                    depth_budget,
                    root_pk,
                    root_cpath,
                    B_active,
                    visited
                );

                ++automatic_refinement_round;
            }

            // special correction:
            // Lickety uses all continuous thresholds, but later things use a binarization
            // in this case, things become optimal 1 lookahead later
            // instead of not clamping as much, we shift the caches down by one
            const bool needs_one_level_proxy_shift =
                !use_restricted_lickety_proxy_() &&
                use_restricted_greedy_proxy_();

            if (
                !proxy_refinement_stopped_early &&
                needs_one_level_proxy_shift &&
                !resource_limit_reached_()
            ) {
                std::cout
                    << "Moving continuous Lickety proxy objectives "
                    << "down one lookahead level.\n";

                shift_proxy_strength_down_one_();

                // same two cache clears as before
                continuous_proxy_completion_cache.clear();
                trie_cache.clear();

                RefineVisited visited;
                visited.reserve(1024);

                RefineGraphDfs(
                    G_min,
                    root,
                    depth_budget,
                    root_pk,
                    root_cpath,
                    B_active,
                    visited
                );

                if (resource_limit_reached_()) {
                    std::cout
                        << "Resource limit reached during corrected "
                        << "proxy refinement.\n";
                }
            }

            // restore the user's original lookahead setting for external consistency
            lookahead_init = old_lookahead_init;

            if (proxy_style == 2) {
                k_at_depth.assign((std::size_t)depth_budget + 1, 1);

                int K = lookahead_init;
                int kk = K;

                for (int dd = depth_budget; dd >= 0; --dd) {
                    k_at_depth[(std::size_t)dd] = std::min(dd, kk);
                    kk = (kk > 1) ? (kk - 1) : K;
                }
            } else {
                k_at_depth.clear();
            }
        }

        allowed_proxy_features = old_allowed_proxy_features;
        restrict_proxy_in_lickety = old_restrict_lickety;
        restrict_proxy_in_depthd_exact = old_restrict_depthd;
        restrict_proxy_in_greedy = old_restrict_greedy;
        anytime_mode_active_ = false;

        result = G_min;
        obj_bound = current_budget;
        best_objective = P_root;
        trained_depth_budget = depth_budget;

        return G_min;
    }

private:

    // varying statistics that may be helpful in extending the budget of the graph to improve approximation quality.
    // these are not part of our contributions
    struct TrieNodeProxyGapSummary {
        double root_gap = 1.0;
        double max_node_gap = 1.0;

        std::size_t nodes_checked = 0;

        const TreeTrieNode* worst_node = nullptr;
        int worst_proxy_objective = -1;
        int worst_minimum_objective = -1;
    };

    void collect_max_trie_node_proxy_gap_(
        const std::shared_ptr<TreeTrieNode>& node,
        const Packed& mask,
        int8_t depth,
        const PathKey& pk,
        const ContinuousPath& cpath,
        std::unordered_set<const TreeTrieNode*>& visited,
        TrieNodeProxyGapSummary& summary
    ) {
        if (!node) {
            return;
        }

        const TreeTrieNode* node_ptr = node.get();

        // a cached TreeTrieNode may be reachable through multiple graph paths.
        // its subproblem/depth is canonical, so evaluate it only once.
        if (!visited.insert(node_ptr).second) {
            return;
        }

        if (
            node->min_objective == std::numeric_limits<int>::max() ||
            node->min_objective <= 0
        ) {
            return;
        }

        const int proxy_objective =
            proxy_objective_for_gap_(
                mask,
                depth,
                pk,
                cpath
            );

        const double node_gap = std::max(
            1.0,
            static_cast<double>(proxy_objective) /
            static_cast<double>(node->min_objective)
        );

        ++summary.nodes_checked;

        if (node_gap > summary.max_node_gap) {
            summary.max_node_gap = node_gap;
            summary.worst_node = node_ptr;
            summary.worst_proxy_objective = proxy_objective;
            summary.worst_minimum_objective =
                node->min_objective;
        }

        if (depth <= 0) {
            return;
        }

        Packed left_mask(n_words);
        Packed right_mask(n_words);

        for (const auto& split : node->splits) {
            if (!split.left || !split.right) {
                continue;
            }

            const int feat = split.feature;

            popcount_and_make_split_words(
                mask.w.data(),
                X_bits[(std::size_t)feat].w.data(),
                left_mask.w.data(),
                right_mask.w.data(),
                n_words,
                tail_mask
            );

            if (!left_mask.any() || !right_mask.any()) {
                continue;
            }

            const PathKey* left_pk = &empty_pk();
            const PathKey* right_pk = &empty_pk();

            PathKey left_pk_local;
            PathKey right_pk_local;

            make_child_pks_if_needed_(
                feat,
                pk,
                left_pk,
                right_pk,
                left_pk_local,
                right_pk_local
            );

            const ContinuousPath* left_cpath = &cpath;
            const ContinuousPath* right_cpath = &cpath;

            ContinuousPath left_cpath_local;
            ContinuousPath right_cpath_local;

            make_child_continuous_paths_if_needed_(
                feat,
                cpath,
                left_cpath,
                right_cpath,
                left_cpath_local,
                right_cpath_local
            );

            const int8_t child_depth =
                static_cast<int8_t>(depth - 1);

            collect_max_trie_node_proxy_gap_(
                split.left,
                left_mask,
                child_depth,
                *left_pk,
                *left_cpath,
                visited,
                summary
            );

            collect_max_trie_node_proxy_gap_(
                split.right,
                right_mask,
                child_depth,
                *right_pk,
                *right_cpath,
                visited,
                summary
            );
        }
    }

    struct ProxyGapSummary {
        double root_gap = 1.0;
        double max_split_gap = 1.0;

        std::size_t distinct_nodes_visited = 0;
        std::size_t splits_checked = 0;

        const TreeTrieNode* worst_node = nullptr;
        int worst_feature = -1;

        int worst_proxy_sum = 0;
        int worst_minimum_sum = 0;
    };

    int proxy_objective_for_gap_(
        const Packed& mask,
        int8_t depth,
        const PathKey& pk,
        const ContinuousPath& cpath
    ) {
        if (depth <= 0 || lookahead_init < 0) {
            return leaf_objective(mask);
        }

        if (lookahead_init == 0) {
            return greedy_proxy_objective_(
                mask,
                depth,
                pk,
                cpath
            );
        }

        int8_t k_here = lookahead_init;

        if (
            proxy_style == 2 &&
            depth >= 0 &&
            depth < static_cast<int>(k_at_depth.size())
        ) {
            k_here = static_cast<int8_t>(
                k_at_depth[static_cast<std::size_t>(depth)]
            );
        } else {
            k_here = std::min<int8_t>(k_here, depth);
        }

        if (proxy_style == 4) {
            return split_algorithm(
                mask,
                depth,
                k_here,
                pk
            );
        }

        return lickety_proxy_objective_(
            mask,
            depth,
            k_here,
            pk,
            cpath
        );
    }

    void extend_result_to_budget_(
        int8_t depth_budget,
        int new_budget,
        const Packed& root,
        const PathKey& root_pk,
        const ContinuousPath& root_cpath,
        const std::vector<int>& all_features
    ) {
        result = construct_trie_extend(
            result,
            root,
            depth_budget,
            new_budget,
            root_pk,
            root_cpath,
            &all_features,
            /*launched_by_anytime=*/false
        );

        if (!result) {
            throw std::runtime_error(
                "construct_trie_extend returned an empty root graph."
            );
        }

        obj_bound = new_budget;
        trained_depth_budget = depth_budget;

        std::cout
            << "Extended minimum objective: "
            << result->min_objective
            << "\n";
    }

    void collect_max_proxy_gap_(
        const std::shared_ptr<TreeTrieNode>& node,
        const Packed& mask,
        int8_t depth,
        const PathKey& pk,
        const ContinuousPath& cpath,
        std::unordered_set<const TreeTrieNode*>& visited,
        ProxyGapSummary& summary
    ) {
        if (!node || depth <= 0) {
            return;
        }

        const TreeTrieNode* node_ptr = node.get();

        // a treetrienode is canonical for a subproblem-depth pair, so once we
        // process it, encountering it through another parent does not give us new splits or new child subproblems.
        
        if (!visited.insert(node_ptr).second) {
            return;
        }

        ++summary.distinct_nodes_visited;

        Packed left_mask(n_words);
        Packed right_mask(n_words);

        for (const SplitNode& split : node->splits) {
            if (!split.left || !split.right) {
                continue;
            }

            const int feat = split.feature;

            split_threshold_bits_(
                mask,
                feat,
                left_mask,
                right_mask
            );

            if (!left_mask.any() || !right_mask.any()) {
                continue;
            }

            const PathKey* left_pk = &empty_pk();
            const PathKey* right_pk = &empty_pk();

            PathKey left_pk_local;
            PathKey right_pk_local;

            make_child_pks_if_needed_(
                feat,
                pk,
                left_pk,
                right_pk,
                left_pk_local,
                right_pk_local
            );

            const ContinuousPath* left_cpath = &cpath;
            const ContinuousPath* right_cpath = &cpath;

            ContinuousPath left_cpath_local;
            ContinuousPath right_cpath_local;

            make_child_continuous_paths_if_needed_(
                feat,
                cpath,
                left_cpath,
                right_cpath,
                left_cpath_local,
                right_cpath_local
            );

            const int8_t child_depth =
                static_cast<int8_t>(depth - 1);

            const int proxy_left =
                proxy_objective_for_gap_(
                    left_mask,
                    child_depth,
                    *left_pk,
                    *left_cpath
                );

            const int proxy_right =
                proxy_objective_for_gap_(
                    right_mask,
                    child_depth,
                    *right_pk,
                    *right_cpath
                );

            const int minimum_left =
                split.left->min_objective;

            const int minimum_right =
                split.right->min_objective;

            if (
                minimum_left == std::numeric_limits<int>::max() ||
                minimum_right == std::numeric_limits<int>::max()
            ) {
                continue;
            }

            const int proxy_sum =
                proxy_left + proxy_right;

            const int minimum_sum =
                minimum_left + minimum_right;

            if (minimum_sum <= 0) {
                throw std::logic_error(
                    "Encountered nonpositive minimum completion objective "
                    "while computing proxy-gap ratios."
                );
            }

            const double split_gap = std::max(
                1.0,
                static_cast<double>(proxy_sum) /
                static_cast<double>(minimum_sum)
            );

            ++summary.splits_checked;

            if (split_gap > summary.max_split_gap) {
                summary.max_split_gap = split_gap;
                summary.worst_node = node_ptr;
                summary.worst_feature = feat;
                summary.worst_proxy_sum = proxy_sum;
                summary.worst_minimum_sum = minimum_sum;
            }

            collect_max_proxy_gap_(
                split.left,
                left_mask,
                child_depth,
                *left_pk,
                *left_cpath,
                visited,
                summary
            );

            collect_max_proxy_gap_(
                split.right,
                right_mask,
                child_depth,
                *right_pk,
                *right_cpath,
                visited,
                summary
            );
        }
    }

    int trie_node_gap_extension_budget_(
        const std::shared_ptr<TreeTrieNode>& root_node,
        const Packed& root_mask,
        int8_t depth_budget,
        int first_budget,
        int current_budget,
        const PathKey& root_pk,
        const ContinuousPath& root_cpath
    ) {
        if (!root_node) {
            throw std::runtime_error(
                "Cannot compute a trie-node-gap extension budget "
                "without a root graph."
            );
        }

        if (
            root_node->min_objective <= 0 ||
            root_node->min_objective ==
                std::numeric_limits<int>::max()
        ) {
            throw std::runtime_error(
                "The root graph has an invalid minimum objective."
            );
        }

        TrieNodeProxyGapSummary summary;

        summary.root_gap = std::max(
            1.0,
            static_cast<double>(best_objective) /
            static_cast<double>(root_node->min_objective)
        );

        std::unordered_set<const TreeTrieNode*> visited;
        visited.reserve(trie_cache.size() * 2 + 16);

        collect_max_trie_node_proxy_gap_(
            root_node,
            root_mask,
            depth_budget,
            root_pk,
            root_cpath,
            visited,
            summary
        );

        const double relative_gap = std::max(
            1.0,
            summary.max_node_gap / summary.root_gap
        );

        // anchor every round to the first-stage budget.
        const long double raw_candidate =
            static_cast<long double>(first_budget) *
            static_cast<long double>(relative_gap);

        if (
            raw_candidate >
            static_cast<long double>(
                std::numeric_limits<int>::max()
            )
        ) {
            throw std::overflow_error(
                "Trie-node-gap extension budget exceeds integer range."
            );
        }

        const int candidate_budget =
            static_cast<int>(std::ceil(raw_candidate));

        std::cout
            << "Trie-node-gap automatic extension:\n"
            << "  First-stage budget: "
            << first_budget
            << "\n"
            << "  Current budget: "
            << current_budget
            << "\n"
            << "  Root proxy objective: "
            << best_objective
            << "\n"
            << "  Root minimum objective: "
            << root_node->min_objective
            << "\n"
            << "  Root proxy-gap ratio: "
            << summary.root_gap
            << "\n"
            << "  Maximum trie-node proxy-gap ratio: "
            << summary.max_node_gap
            << "\n"
            << "  Relative ratio: "
            << relative_gap
            << "\n"
            << "  Candidate budget: "
            << candidate_budget
            << "\n"
            << "  Trie nodes checked: "
            << summary.nodes_checked
            << "\n";

        if (summary.worst_node != nullptr) {
            std::cout
                << "  Worst node proxy objective: "
                << summary.worst_proxy_objective
                << "\n"
                << "  Worst node minimum objective: "
                << summary.worst_minimum_objective
                << "\n";
        }

        return std::max(first_budget, candidate_budget);
    }

    int split_gap_extension_budget_(
        const std::shared_ptr<TreeTrieNode>& root_node,
        const Packed& root_mask,
        int8_t depth_budget,
        int first_budget,
        int current_budget,
        const PathKey& root_pk,
        const ContinuousPath& root_cpath
    ) {
        if (!root_node) {
            throw std::runtime_error(
                "Cannot compute a split-gap extension budget "
                "without a root graph."
            );
        }

        if (
            root_node->min_objective <= 0 ||
            root_node->min_objective ==
                std::numeric_limits<int>::max()
        ) {
            throw std::runtime_error(
                "The root graph has an invalid minimum objective."
            );
        }

        ProxyGapSummary summary;

        summary.root_gap = std::max(
            1.0,
            static_cast<double>(best_objective) /
            static_cast<double>(root_node->min_objective)
        );

        std::unordered_set<const TreeTrieNode*> visited;
        visited.reserve(trie_cache.size() * 2 + 16);

        collect_max_proxy_gap_(
            root_node,
            root_mask,
            depth_budget,
            root_pk,
            root_cpath,
            visited,
            summary
        );

        const double relative_gap = std::max(
            1.0,
            summary.max_split_gap / summary.root_gap
        );

        // use the original first-stage budget,
        // not the current extended budget.
        const long double raw_candidate =
            static_cast<long double>(first_budget) *
            static_cast<long double>(relative_gap);

        if (
            raw_candidate >
            static_cast<long double>(
                std::numeric_limits<int>::max()
            )
        ) {
            throw std::overflow_error(
                "Split-gap extension budget exceeds integer range."
            );
        }

        const int candidate_budget =
            static_cast<int>(std::ceil(raw_candidate));

        std::cout
            << "Split-gap automatic extension:\n"
            << "  First-stage budget: "
            << first_budget
            << "\n"
            << "  Current budget: "
            << current_budget
            << "\n"
            << "  Root proxy objective: "
            << best_objective
            << "\n"
            << "  Root minimum objective: "
            << root_node->min_objective
            << "\n"
            << "  Root proxy-gap ratio: "
            << summary.root_gap
            << "\n"
            << "  Maximum split proxy-gap ratio: "
            << summary.max_split_gap
            << "\n"
            << "  Relative ratio: "
            << relative_gap
            << "\n"
            << "  Candidate budget: "
            << candidate_budget
            << "\n"
            << "  Splits checked: "
            << summary.splits_checked
            << "\n";

        if (summary.worst_node != nullptr) {
            std::cout
                << "  Worst split feature: "
                << summary.worst_feature
                << "\n"
                << "  Worst split proxy sum: "
                << summary.worst_proxy_sum
                << "\n"
                << "  Worst split minimum sum: "
                << summary.worst_minimum_sum
                << "\n";
        }

        return std::max(first_budget, candidate_budget);
    }

    // the following operations are largely existing and low-level. not part of our contribution.
    static inline uint8_t pred_to_mask_(int pred) {
        if (pred == 0) return uint8_t(1);
        if (pred == 1) return uint8_t(2);

        if (pred >= 0 && pred < 8) {
            return uint8_t(1u << pred);
        }

        throw std::runtime_error("Prediction label too large for uint8_t mask.");
    }

    static inline uint64_t class_to_mask_(int prediction) {
        if (prediction < 0 || prediction >= 64) {
            throw std::runtime_error(
                "Reachable class masks support class labels 0 through 63."
            );
        }

        return uint64_t(1) << prediction;
    }

    ReachableActions
    reachable_actions_for_training_sample_rec_(
        const std::shared_ptr<TreeTrieNode>& node,
        int sample_idx
    ) const {
        ReachableActions out;

        if (!node) {
            return out;
        }

        for (const auto& leaf : node->leaves) {
            if (
                leaf.prediction ==
                DEFER_PREDICTION
            ) {
                out.can_defer = true;
            } else {
                out.class_mask |=
                    class_to_mask_(
                        leaf.prediction
                    );
            }
        }

        for (const auto& split : node->splits) {
            const bool go_left =
                training_value_(
                    sample_idx,
                    split.feature
                );

            const auto& child =
                go_left
                    ? split.left
                    : split.right;

            const ReachableActions child_actions =
                reachable_actions_for_training_sample_rec_(
                    child,
                    sample_idx
                );

            out.class_mask |=
                child_actions.class_mask;

            out.can_defer =
                out.can_defer ||
                child_actions.can_defer;
        }

        return out;
    }

    bool training_value_(int sample_idx, int feat) const {
        if (sample_idx < 0 || sample_idx >= n_samples) {
            throw std::runtime_error("training_value_ got out-of-range sample index.");
        }
        if (feat < 0 || feat >= n_features) {
            throw std::runtime_error("training_value_ got out-of-range feature index.");
        }

        const uint64_t bit = 1ULL << (sample_idx & 63);
        return (X_bits[(size_t)feat].w[(size_t)(sample_idx >> 6)] & bit) != 0;
    }

    uint8_t reachable_prediction_mask_for_training_sample_rec_(
        const std::shared_ptr<TreeTrieNode>& node,
        int sample_idx
    ) const {
        if (!node) return 0;

        uint8_t mask = 0;

        // any leaf option at this OR node is reachable.
        for (const auto& leaf : node->leaves) {
            mask |= pred_to_mask_(leaf.prediction);

            // early exit for binary case.
            if ((mask & uint8_t(1)) && (mask & uint8_t(2))) {
                return mask;
            }
        }

        // for every split option, path this sample down the branch it follows.
        for (const auto& s : node->splits) {
            const bool go_left = training_value_(sample_idx, s.feature);

            const std::shared_ptr<TreeTrieNode>& child =
                go_left ? s.left : s.right;

            mask |= reachable_prediction_mask_for_training_sample_rec_(
                child,
                sample_idx
            );

            if ((mask & uint8_t(1)) && (mask & uint8_t(2))) {
                return mask;
            }
        }

        return mask;
    }

    // has the anytime algorithm activated all thresholds?
    bool active_contains_all_threshold_columns_(
        const std::vector<int>& B_active
    ) const {
        if (continuous_starts.empty()) {
            return true;
        }

        const int first_cont = first_continuous_feature_();

        for (int f = first_cont; f < n_features; ++f) {
            if (!std::binary_search(B_active.begin(), B_active.end(), f)) {
                return false;
            }
        }

        return true;
    }

    // we select new thresholds by taking the midpoint between adjacent active thresholds. 
    // we do this for each adjacent active threshold. 
    // this corresponds to our theoretical guarantees discussed in the appendix, and what we say is our default behavior in the algorithm description.
    std::vector<int> SelectNewThresholds(
        const std::vector<int>& B_active,
        const std::vector<int>& continuous_starts,
        int refinement_width
    ) const {
        if (refinement_width <= 0 || continuous_starts.empty()) return {};

        std::vector<int> active = B_active;
        // sort_unique_ints_inplace_(active);

        std::vector<int> out;

        for (int g = 0; g < (int)continuous_starts.size(); ++g) {
            const int start = continuous_starts[(std::size_t)g];
            const int end = continuous_group_end_(g); // exclusive

            std::vector<int> A;
            for (int f = start; f < end; ++f) {
                if (std::binary_search(active.begin(), active.end(), f)) {
                    A.push_back(f);
                }
            }

            // if (A.empty()) continue; // if A is empty, then we want to start phasing in using those thresholds
            // in this case, anchors = {start - 1, end};, so we're taking evenly spaced thresholds in threshold index space based on our parameter

            // also refine before the first active threshold and after the last one.
            std::vector<int> anchors;
            anchors.reserve(A.size() + 2);
            anchors.push_back(start - 1);
            anchors.insert(anchors.end(), A.begin(), A.end());
            anchors.push_back(end);

            for (int t = 0; t + 1 < (int)anchors.size(); ++t) {
                const int lo = anchors[(std::size_t)t];
                const int hi = anchors[(std::size_t)(t + 1)];

                const int gap = hi - lo - 1;
                if (gap <= 0) continue;

                if (gap <= refinement_width) {
                    for (int f = lo + 1; f <= hi - 1; ++f) {
                        out.push_back(f);
                    }
                } else {
                    for (int k = 1; k <= refinement_width; ++k) {
                        const int f = lo + (int)std::llround(
                            (double)k * (double)(hi - lo) /
                            (double)(refinement_width + 1)
                        );

                        if (f > lo && f < hi) {
                            out.push_back(f);
                        }
                    }
                }
            }
        }

        sort_unique_ints_inplace_(out);
        return out;
    }


    template <typename T>
    static void validate_rectangular_matrix_(
        const std::vector<std::vector<T>>& X,
        const std::string& name
    ) {
        if (X.empty()) return;

        const std::size_t p = X[0].size();
        for (std::size_t i = 1; i < X.size(); ++i) {
            if (X[i].size() != p) {
                throw std::runtime_error(name + " must be rectangular.");
            }
        }
    }

    static std::vector<double> sorted_unique_values_(
        const std::vector<std::vector<double>>& X_num_row_major,
        int col
    ) {
        std::vector<double> vals;
        vals.reserve(X_num_row_major.size());

        for (std::size_t i = 0; i < X_num_row_major.size(); ++i) {
            const double v = X_num_row_major[i][(std::size_t)col];

            if (std::isnan(v)) {
                throw std::runtime_error("X_num contains NaN.");
            }

            vals.push_back(v);
        }

        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        return vals;
    }

    // how many samples do these differ by?
    static int hamming_distance_binary_column_(
        const std::vector<bool>& a,
        const std::vector<uint8_t>& b
    ) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Hamming distance column sizes do not match.");
        }

        int d = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const bool bv = (b[i] != 0);
            if (a[i] != bv) ++d;
        }
        return d;
    }

    static void sort_unique_ints_inplace_(std::vector<int>& xs) {
        std::sort(xs.begin(), xs.end());
        xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    }

    void clear_fit_state_() {
        result.reset();

        X_bits.clear();
        Y_bits.clear();
        continuous_starts.clear();

        use_deferral = false;
        eta_defer = 0.0;
        BBwrong = Packed();

        greedy_cache.clear();
        lickety_cache_k2.clear();
        lickety_cache_kla.clear();
        trie_cache.clear();
        greedy_first_split_cache.clear();
        anytime_lickety_first_split_cache.clear();
        anytime_mode_active_ = false;
        single_tree_refined_override_.reset();

        continuous_proxy_completion_cache.clear();
        continuous_proxy_completion_cache.rehash(0);
        

        mask_ids = MaskIdTable();
        lit_ids = LitIdTable();
        fingerprint128_ids = Fingerprint128IdTable();

        n_samples = 0;
        n_features = 0;
        n_words = 0;
        tail_mask = ~0ULL;
        gamma = 0;
        trained_depth_budget = -1;
        best_objective = 0;
        obj_bound = 0;
        num_classes = 0;
        k_at_depth.clear();
    }

    inline bool mask_has_row_(const Packed& mask, int row) const {
        const int w = row >> 6;
        const int b = row & 63;
        return ((mask.w[(std::size_t)w] >> b) & 1ULL) != 0ULL;
    }
    
    // we ablate this choice in the appendix
    struct NumericalGreedyState {
        // one sorted row list per numerical continuous group.
        // sorted_idx_by_num[g] is sorted by numerical_X_cols_for_greedy[g].
        std::vector<std::vector<int>> sorted_idx_by_num;
    };

    // a wrapper equivalent of ContinuousRSet
    // as seen just below, if we need to extend a graph to a bigger budget, we switch over to this construct_trie_extend method, which is an exact implementation of ContinuousRSet
    shared_ptr<TreeTrieNode> construct_trie(const Packed& mask, int8_t depth, int budget, const PathKey& pk, const ContinuousPath& cpath = empty_continuous_path(), const std::vector<int>* active_features = nullptr) {
        // testing something
        auto node = std::make_shared<TreeTrieNode>();
        // force construct_trie_extend to treat this as an unsolved node.
        node->budget = -1;

        return construct_trie_extend(
            node,
            mask,
            depth,
            budget,
            pk,
            cpath,
            active_features
        );
    }

    // we skip this method for simplicity. we note that this does not call initandprune style logic
    // because we are in construct_trie, not construct_trie_extend, and thus we should not have any threshold-to-proxy-completion maps
    // if we had those maps, we would have been here with a different budget, and that budget would have been smaller, so we now would be extending the graph
    // if we were extending the graph, we would not be calling this because this is only called from construct_trie
    void enumerate_active_continuous_feature_for_trie_restricted(
        shared_ptr<TreeTrieNode>& node,
        const Packed& mask,
        int8_t depth,
        int budget,
        const PathKey& pk,
        ContinuousPath& cpath,
        int8_t k_here,
        int start_idx,
        int end_idx,
        Packed& L,
        Packed& R,
        const std::vector<int>* active_features
    ) {
        if (!active_features || start_idx >= end_idx) return;

        const auto active_lo = std::lower_bound(
            active_features->begin(),
            active_features->end(),
            start_idx
        );

        const auto active_hi = std::lower_bound(
            active_features->begin(),
            active_features->end(),
            end_idx
        );

        if (active_lo == active_hi) return;

        // create a non-owning view of that subarray
        std::span<const int> active_thresholds(
            &(*active_lo),
            (std::size_t)std::distance(active_lo, active_hi)
        );


        if (active_thresholds.empty()) return;

        const uint64_t kmask = key_of_subproblem(mask, pk);
        KContProxy cache_key{kmask, depth, start_idx};
        ProxyCompletionTree& evaluated =
            continuous_proxy_completion_cache[cache_key];

        // ContinuousPath q_cpath = materialize_continuous_path_(cpath);
        if ((int)cpath.size() != (int)continuous_starts.size()) {
            initialize_continuous_path_(cpath);
        }

        std::deque<std::pair<int,int>> Q;
        Q.push_back({0, (int)active_thresholds.size() - 1});

        while (!Q.empty()) {
            auto [i, j] = Q.front();
            Q.pop_front();

            if (i > j) continue;

            const int mid = i + ((j - i) >> 1);
            const int feat = active_thresholds[(std::size_t)mid];

            split_threshold_bits_(mask, feat, L, R);

            const bool left_empty = !L.any();
            const bool right_empty = !R.any();

            if (left_empty || right_empty) {
                if (left_empty && !right_empty) {
                    constrain_continuous_false_branch_(cpath, feat);
                    if (mid + 1 <= j) {
                        Q.push_back({mid + 1, j});
                    }
                } else if (!left_empty && right_empty) {
                    constrain_continuous_true_branch_(cpath, feat);
                    if (i <= mid - 1) {
                        Q.push_back({i, mid - 1});
                    }
                }
                continue;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                feat,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                feat,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            int lossL;
            int lossR;

            // there is little need to look at the evaluated map here, we could just go to proxy completions
            // auto it_eval = evaluated.find(feat);

            if (false) {
                lossL = 0;
                lossR = 0;
            } else {
                if (lookahead_init < 0) {
                    lossL = leaf_objective(L);
                } else if (lookahead_init == 0) {
                    lossL = greedy_proxy_objective_(
                        L,
                        depth - 1,
                        *pkLp,
                        *cpathLp
                    );
                } else {
                    if (proxy_style == 4) {
                        lossL = split_algorithm(
                            L,
                            depth - 1,
                            k_here,
                            *pkLp
                        );
                    } else {
                        lossL = lickety_proxy_objective_(
                            L,
                            depth - 1,
                            k_here,
                            *pkLp,
                            *cpathLp
                        );
                    }
                }

                if (lookahead_init < 0) {
                    lossR = leaf_objective(R);
                } else if (lookahead_init == 0) {
                    lossR = greedy_proxy_objective_(
                        R,
                        depth - 1,
                        *pkRp,
                        *cpathRp
                    );
                } else {
                    if (proxy_style == 4) {
                        lossR = split_algorithm(
                            R,
                            depth - 1,
                            k_here,
                            *pkRp
                        );
                    } else {
                        lossR = lickety_proxy_objective_(
                            R,
                            depth - 1,
                            k_here,
                            *pkRp,
                            *cpathRp
                        );
                    }
                }

                evaluated[feat] = {lossL, lossR};
            }

            const int total_proxy = lossL + lossR;

            bool feasible;
            if (!rule_list_mode) {
                feasible = (total_proxy <= budget);
            } else {
                feasible = !(lossL > budget - gamma && lossR > budget - gamma);
            }

            if (feasible) {
                std::pair<
                    std::shared_ptr<TreeTrieNode>,
                    std::shared_ptr<TreeTrieNode>
                > LR;

                if (rule_list_mode || use_multipass) {
                    LR = solve_siblings_extend(
                        nullptr,
                        nullptr,
                        lossL,
                        lossR,
                        L,
                        R,
                        budget,
                        depth,
                        *pkLp,
                        *pkRp,
                        *cpathLp,
                        *cpathRp,
                        active_features
                    );
                } else {
                    LR = symmetric_single_pass(
                        lossL,
                        lossR,
                        L,
                        R,
                        budget,
                        depth,
                        *pkLp,
                        *pkRp,
                        *cpathLp,
                        *cpathRp
                    );
                }

               if (
                    !children_have_feasible_pair_(
                        LR.first,
                        LR.second,
                        budget
                    )
                ) {
                    if (resource_limit_reached_()) {
                        return;
                    }
                } else {
                    node->add_split(feat, LR.first, LR.second);
                }

                if (i <= mid - 1) {
                    Q.push_back({i, mid - 1});
                }

                if (mid + 1 <= j) {
                    Q.push_back({mid + 1, j});
                }

                continue;
            }

            const int delta = total_proxy - budget;

            if (delta <= 0) {
                if (i <= mid - 1) Q.push_back({i, mid - 1});
                if (mid + 1 <= j) Q.push_back({mid + 1, j});
                continue;
            }

            // moving left from feat: find the farthest active threshold on the left
            // that is at least delta samples away from feat.
            const int a = tighten_upper_bound_active_thresholds_(
                mask,
                active_thresholds,
                mid,
                i,
                mid - 1,
                delta
            );

            // moving right from feat: find the first active threshold on the right
            // that is at least delta samples away from feat.
            const int b = tighten_lower_bound_active_thresholds_(
                mask,
                active_thresholds,
                mid,
                mid + 1,
                j,
                delta
            );

            if (i <= a) {
                Q.push_back({i, a});
            }

            if (b <= j) {
                Q.push_back({b, j});
            }
        }
    }

    int tighten_lower_bound_active_thresholds_(
        const Packed& mask,
        std::span<const int> active_thresholds,
        int anchor_pos,
        int lo_pos,
        int hi_pos,
        int delta
    ) const {
        if (delta <= 0) return lo_pos;
        if (lo_pos > hi_pos) return hi_pos + 1;

        const int anchor_feat = active_thresholds[(std::size_t)anchor_pos];

        // local probe first.
        const int probe_end = std::min(hi_pos, lo_pos + 1);
        for (int p = lo_pos; p <= probe_end; ++p) {
            const int feat = active_thresholds[(std::size_t)p];
            if (bit_distance_at_least_(mask, anchor_feat, feat, delta)) {
                return p;
            }
        }

        int lo = probe_end + 1;
        int hi = hi_pos;
        int ans = hi_pos + 1;

        while (lo <= hi) {
            const int mid = lo + ((hi - lo) >> 1);
            const int feat = active_thresholds[(std::size_t)mid];

            if (bit_distance_at_least_(mask, anchor_feat, feat, delta)) {
                ans = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }

        return ans;
    }

    int tighten_upper_bound_active_thresholds_(
        const Packed& mask,
        std::span<const int> active_thresholds,
        int anchor_pos,
        int lo_pos,
        int hi_pos,
        int delta
    ) const {
        if (delta <= 0) return hi_pos;
        if (lo_pos > hi_pos) return lo_pos - 1;

        const int anchor_feat = active_thresholds[(std::size_t)anchor_pos];

        // local probe first.
        const int probe_start = std::max(lo_pos, hi_pos - 1);
        for (int p = hi_pos; p >= probe_start; --p) {
            const int feat = active_thresholds[(std::size_t)p];
            if (bit_distance_at_least_(mask, feat, anchor_feat, delta)) {
                return p;
            }
        }

        int lo = lo_pos;
        int hi = probe_start - 1;
        int ans = lo_pos - 1;

        while (lo <= hi) {
            const int mid = lo + ((hi - lo) >> 1);
            const int feat = active_thresholds[(std::size_t)mid];

            if (bit_distance_at_least_(mask, feat, anchor_feat, delta)) {
                ans = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        return ans;
    }

    int raw_continuous_start_for_threshold_(int start_idx) const {
        auto it = std::upper_bound(
            continuous_starts.begin(),
            continuous_starts.end(),
            start_idx
        );

        if (it == continuous_starts.begin()) {
            throw std::runtime_error(
                "start_idx is before the first continuous feature group."
            );
        }

        --it;
        return *it;
    }

    // not just does it exist, but does it have either leaves or a split. this will guarantee every node in the graph leads to a solution by induction
    static bool node_has_solution_(
        const std::shared_ptr<TreeTrieNode>& node
    ) {
        return node &&
            (
                !node->leaves.empty() ||
                !node->splits.empty()
            );
    }

    // we actually need to guarantee that there is a solution within the budget
    static bool children_have_feasible_pair_(
        const std::shared_ptr<TreeTrieNode>& left,
        const std::shared_ptr<TreeTrieNode>& right,
        int parent_budget
    ) {
        if (!node_has_solution_(left) ||
            !node_has_solution_(right)) {
            return false;
        }

        if (
            left->min_objective ==
                std::numeric_limits<int>::max() ||
            right->min_objective ==
                std::numeric_limits<int>::max()
        ) {
            return false;
        }

        return left->min_objective <=
            parent_budget - right->min_objective;
    }

    // this algorithm is a combined version of InitAndPrune and EnumContFeature
    // it collects the acurrently active thresholds, gets E, loops over all cached threshold completions, if it is feasible solve or resolve it,
    //  otherwise tighten global search bounds, prune the neighborhood, take the complement, 
    // ..... do the queue logic, and so on. a full implementation of those two methods.
    void enumerate_continuous_feature_for_trie_extend_restricted(
        shared_ptr<TreeTrieNode>& node,
        const Packed& mask,
        int8_t depth,
        int budget,
        const PathKey& pk,
        ContinuousPath& cpath,
        int8_t k_here,
        int start_idx,
        int end_idx,
        Packed& L,
        Packed& R,
        std::unordered_set<int>& already_split,
        const std::vector<int>* active_features
    ) {
        if (!node || !active_features || start_idx >= end_idx) return;

        // for (int f : *active_features) {
        //     if (f >= start_idx && f < end_idx) {
        //         active_thresholds.push_back(f);
        //     }
        // }

        const auto active_lo = std::lower_bound(
            active_features->begin(),
            active_features->end(),
            start_idx
        );

        const auto active_hi = std::lower_bound(
            active_features->begin(),
            active_features->end(),
            end_idx
        );

        const bool have_active_thresholds = active_lo != active_hi;

        std::span<const int> active_thresholds;
        if (have_active_thresholds) {
            active_thresholds = std::span<const int>(
                &(*active_lo),
                (std::size_t)std::distance(active_lo, active_hi)
            );
        }

        const int raw_start_idx = raw_continuous_start_for_threshold_(start_idx);

        const uint64_t kmask = key_of_subproblem(mask, pk);
        KContProxy cache_key{kmask, depth, raw_start_idx};

        ProxyCompletionTree& evaluated =
            continuous_proxy_completion_cache[cache_key];

        // ContinuousPath q_cpath = materialize_continuous_path_(cpath);
        if ((int)cpath.size() != (int)continuous_starts.size()) {
            initialize_continuous_path_(cpath);
        }

        const int anytime_lk1_feat =
            lookup_anytime_lickety_first_split_(mask, depth, pk);

        const bool anytime_lk1_feat_in_this_group =
            anytime_lk1_feat >= start_idx && anytime_lk1_feat < end_idx;

        const bool anytime_lk1_feat_already_active =
            anytime_lk1_feat_in_this_group &&
            feature_in_sorted_vector_(*active_features, anytime_lk1_feat);
            

        std::deque<std::pair<int,int>> Q;

        // global bounds
        int M_L = 0; 
        int M_R = (int)active_thresholds.size() - 1;

        // invariant/assumption: only call on things within budget
        auto resolve_threshold = [&](
            int feat,
            int lossL,
            int lossR
        ) -> bool {
            split_threshold_bits_(mask, feat, L, R);

            if (!L.any() || !R.any()) {
                return false;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                feat,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                feat,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            auto it_split = std::find_if(
                node->splits.begin(),
                node->splits.end(),
                [&](const SplitNode& s) {
                    return s.feature == feat;
                }
            );

            const bool exists = (it_split != node->splits.end());

            std::shared_ptr<TreeTrieNode> old_left = nullptr;
            std::shared_ptr<TreeTrieNode> old_right = nullptr;

            if (exists) {
                old_left = it_split->left;
                old_right = it_split->right;

                // if (old_left) {
                //     lossL = old_left->min_objective;
                // }

                // if (old_right) {
                //     lossR = old_right->min_objective;
                // }
            }

            std::pair<
                std::shared_ptr<TreeTrieNode>,
                std::shared_ptr<TreeTrieNode>
            > LR;

            if (exists || rule_list_mode || use_multipass) {
                LR = solve_siblings_extend(
                    old_left,
                    old_right,
                    lossL,
                    lossR,
                    L,
                    R,
                    budget,
                    depth,
                    *pkLp,
                    *pkRp,
                    *cpathLp,
                    *cpathRp,
                    active_features,
                    true
                );
            } else {
                LR = symmetric_single_pass(
                    lossL,
                    lossR,
                    L,
                    R,
                    budget,
                    depth,
                    *pkLp,
                    *pkRp,
                    *cpathLp,
                    *cpathRp
                );
            }

            if (
                !children_have_feasible_pair_(
                    LR.first,
                    LR.second,
                    budget
                )
            ) {
                return false;
            }

            if (exists) {
                it_split->left = LR.first;
                it_split->right = LR.second;

                const int min_sum =
                    LR.first->min_objective + LR.second->min_objective;

                if (min_sum < node->min_objective) {
                    node->min_objective = min_sum;
                }
            } else {
                node->add_split(feat, LR.first, LR.second);
                already_split.insert(feat);
            }

            return true;
        };

        // if continuous Lickety k=1 chose a threshold outside the active set,
        // try to add/extend that exact split. 
        if (anytime_lk1_feat_in_this_group && !anytime_lk1_feat_already_active) {
            const int feat = anytime_lk1_feat;

            split_threshold_bits_(mask, feat, L, R);

            if (L.any() && R.any()) {
                const PathKey* pkLp = &empty_pk();
                const PathKey* pkRp = &empty_pk();

                PathKey pkL_local;
                PathKey pkR_local;

                make_child_pks_if_needed_(
                    feat,
                    pk,
                    pkLp,
                    pkRp,
                    pkL_local,
                    pkR_local
                );

                const ContinuousPath* cpathLp = &cpath;
                const ContinuousPath* cpathRp = &cpath;

                ContinuousPath cpathL_local;
                ContinuousPath cpathR_local;

                make_child_continuous_paths_if_needed_(
                    feat,
                    cpath,
                    cpathLp,
                    cpathRp,
                    cpathL_local,
                    cpathR_local
                );

                const int lossL = proxy_completion_objective_(
                    L,
                    depth - 1,
                    k_here,
                    *pkLp,
                    *cpathLp
                );

                const int lossR = proxy_completion_objective_(
                    R,
                    depth - 1,
                    k_here,
                    *pkRp,
                    *cpathRp
                );

                evaluated[feat] = {lossL, lossR};

                const int total_proxy = lossL + lossR;

                const bool feasible =
                    !rule_list_mode
                        ? (total_proxy <= budget)
                        : !(lossL > budget - gamma && lossR > budget - gamma);

                if (feasible) {
                    const bool resolved =
                        resolve_threshold(feat, lossL, lossR);
                    // if it failed (probably becauase of resources), check if we are out of resources, and then stop
                    if (!resolved && resource_limit_reached_()) {
                        return;
                    }
                }
            }
        }

        if (active_thresholds.empty()) {
            return;
        }

        std::map<int,int> pruned_intervals;

        // we maintain pruned_intervals = {{3, 7}, {12, 15} };
        // meaning we pruned 3,4,5,6,7 and 12,13,14,15
        // we need to add lo,hi to this, merging it with previous or after
        auto add_pruned_interval = [&](
            int lo,
            int hi
        ) {
            if (lo > hi) return;

            lo = std::max(lo, 0);
            hi = std::min(hi, (int)active_thresholds.size() - 1);

            if (lo > hi) return;

            auto it = pruned_intervals.upper_bound(lo);

            if (it != pruned_intervals.begin()) {
                auto prev = std::prev(it);

                if (prev->second + 1 >= lo) {
                    lo = std::min(lo, prev->first);
                    hi = std::max(hi, prev->second);
                    it = pruned_intervals.erase(prev);
                }
            }

            while (it != pruned_intervals.end() && it->first <= hi + 1) {
                hi = std::max(hi, it->second);
                it = pruned_intervals.erase(it);
            }

            pruned_intervals[lo] = hi;
        };

        auto it0 = evaluated.lower_bound(start_idx);
        // going over the red black tree, getting proxy completions, if something is feasible, 
        // it either solves it for the first time or resolves it (pruning that index too).
        // if it isn't feasible, it finds the pruned interval
        for (auto it = it0; it != evaluated.end() && it->first < end_idx; ++it) {
            const int feat = it->first;

            auto pos_it = std::lower_bound(
                active_thresholds.begin(),
                active_thresholds.end(),
                feat
            );

            if (pos_it == active_thresholds.end() || *pos_it != feat) {
                continue;
            }

            const int pos =
                (int)std::distance(active_thresholds.begin(), pos_it);

            const int lossL = it->second.first;
            const int lossR = it->second.second;
            const int total_proxy = lossL + lossR;

            bool feasible;
            if (!rule_list_mode) {
                feasible = (total_proxy <= budget);
            } else {
                feasible = !(lossL > budget - gamma && lossR > budget - gamma);
            }

            
            if (feasible) {
                const bool resolved =
                    resolve_threshold(feat, lossL, lossR);

                // similarly, if it failed, we're probably out of resources, if so, stop
                if (!resolved && resource_limit_reached_()) {
                    return;
                }

                add_pruned_interval(pos, pos);
                continue;
            }

            if (!rule_list_mode) {
                if (lossL == gamma) { 
                    M_L = std::max(M_L, pos + 1); 
                } 
                if (lossR == gamma) { 
                    M_R = std::min(M_R, pos - 1); 
                } 
            }

            const int delta = total_proxy - budget;

            if (delta <= 0) {
                continue;
            }

            const int a = tighten_upper_bound_active_thresholds_( mask, active_thresholds, pos, M_L, pos - 1, delta ); 
            const int b = tighten_lower_bound_active_thresholds_( mask, active_thresholds, pos, pos + 1, M_R, delta );

            add_pruned_interval(a + 1, b - 1);
        }

        // makes the complement of the pruned intervals and uses it to initialize Q
        { 
            int cur = M_L; 
            const int last = M_R; 
            for (const auto& kv : pruned_intervals) { 
                const int lo = kv.first; 
                const int hi = kv.second; 
                if (cur <= lo - 1) { 
                    Q.push_back({cur, lo - 1}); 
                } 
                cur = std::max(cur, hi + 1); 
                if (cur > last) break; 
            } 
            if (cur <= last) { 
                Q.push_back({cur, last}); 
            } 
        }

        while (!Q.empty()) {
            auto [i, j] = Q.front(); 
            Q.pop_front(); 
            i = std::max(i, M_L); 
            j = std::min(j, M_R); 
            if (i > j) continue;
            const int mid = i + ((j - i) >> 1);
            const int feat = active_thresholds[(std::size_t)mid];

            // should never happen unless we don't story proxy completions or something, or do the chain logic
            if (already_split.count(feat)) {
                auto it_split = std::find_if(
                    node->splits.begin(),
                    node->splits.end(),
                    [&](const SplitNode& s) {
                        return s.feature == feat;
                    }
                );

                if (it_split != node->splits.end()) {
                    const int lossL = it_split->left
                        ? it_split->left->min_objective
                        : std::numeric_limits<int>::max();

                    const int lossR = it_split->right
                        ? it_split->right->min_objective
                        : std::numeric_limits<int>::max();


                    // same deal
                    const bool resolved =
                        resolve_threshold(feat, lossL, lossR);

                    if (!resolved && resource_limit_reached_()) {
                        return;
                    }
                }

                if (i <= mid - 1) {
                    Q.push_back({i, mid - 1});
                }

                if (mid + 1 <= j) {
                    Q.push_back({mid + 1, j});
                }

                continue;
            }

            split_threshold_bits_(mask, feat, L, R);

            const bool left_empty = !L.any();
            const bool right_empty = !R.any();

            if (left_empty || right_empty) {
                if (left_empty && !right_empty) {
                    constrain_continuous_false_branch_(cpath, feat);
                    M_L = std::max(M_L, mid + 1);
                    if (mid + 1 <= j) {
                        Q.push_back({mid + 1, j});
                    }
                } else if (!left_empty && right_empty) {
                    constrain_continuous_true_branch_(cpath, feat);
                    M_R = std::min(M_R, mid - 1);
                    if (i <= mid - 1) {
                        Q.push_back({i, mid - 1});
                    }
                }

                continue;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                feat,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                feat,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            int lossL;
            int lossR;

            // auto it_eval = evaluated.find(feat);

            // if (it_eval != evaluated.end()) {
            //     lossL = it_eval->second.first;
            //     lossR = it_eval->second.second;
            if (lookahead_init < 0) {
                lossL = leaf_objective(L);
            } else if (lookahead_init == 0) {
                lossL = greedy_proxy_objective_(
                    L,
                    depth - 1,
                    *pkLp,
                    *cpathLp
                );
            } else {
                if (proxy_style == 4) {
                    lossL = split_algorithm(
                        L,
                        depth - 1,
                        k_here,
                        *pkLp
                    );
                } else {
                    lossL = lickety_proxy_objective_(
                        L,
                        depth - 1,
                        k_here,
                        *pkLp,
                        *cpathLp
                    );
                }
            }

            if (lookahead_init < 0) {
                lossR = leaf_objective(R);
            } else if (lookahead_init == 0) {
                lossR = greedy_proxy_objective_(
                    R,
                    depth - 1,
                    *pkRp,
                    *cpathRp
                );
            } else {
                if (proxy_style == 4) {
                    lossR = split_algorithm(
                        R,
                        depth - 1,
                        k_here,
                        *pkRp
                    );
                } else {
                    lossR = lickety_proxy_objective_(
                        R,
                        depth - 1,
                        k_here,
                        *pkRp,
                        *cpathRp
                    );
                }
            }

            evaluated[feat] = {lossL, lossR};
        

            const int total_proxy = lossL + lossR;

            bool feasible;
            if (!rule_list_mode) {
                feasible = (total_proxy <= budget);
            } else {
                feasible = !(lossL > budget - gamma && lossR > budget - gamma);
            }

            if (feasible) {
                const bool resolved = resolve_threshold(feat, lossL, lossR);

                // same thing
                if (!resolved && resource_limit_reached_()) {
                    return;
                }

                if (i <= mid - 1) {
                    Q.push_back({i, mid - 1});
                }

                if (mid + 1 <= j) {
                    Q.push_back({mid + 1, j});
                }

                continue;
            }

            if (!rule_list_mode) { 
                if (lossL == gamma) { 
                    M_L = std::max(M_L, mid + 1); 
                } 
                if (lossR == gamma) { 
                    M_R = std::min(M_R, mid - 1); 
                } 
            }

            const int delta = total_proxy - budget;

            if (delta <= 0) {
                if (i <= mid - 1) {
                    Q.push_back({i, mid - 1});
                }

                if (mid + 1 <= j) {
                    Q.push_back({mid + 1, j});
                }

                continue;
            }

            // no need to edit the calls here because we have i = std::max(i, M_L); j = std::min(j, M_R);
            const int a = tighten_upper_bound_active_thresholds_(
                mask,
                active_thresholds,
                mid,
                i,
                mid - 1,
                delta
            );

            const int b = tighten_lower_bound_active_thresholds_(
                mask,
                active_thresholds,
                mid,
                mid + 1,
                j,
                delta
            );

            if (i <= a) {
                Q.push_back({i, a});
            }

            if (b <= j) {
                Q.push_back({b, j});
            }
        }
    }

    using RefineVisited =
        std::unordered_set<TreeTrieNode*>;

    // an implementation of RefineGraph from the paper (appendix). as in the algorithm, it takes in the visited set, which starts out empty, so this recursive graph procedure can add to it.
    void RefineGraphDfs(
        std::shared_ptr<TreeTrieNode>& G,
        const Packed& mask,
        int8_t depth,
        const PathKey& pk,
        const ContinuousPath& cpath,
        const std::vector<int>& B_active,
        RefineVisited& visited
    ) {
        
        if (!G) return;

        TreeTrieNode* node_ptr = G.get();

        // insert returns {iterator, was_inserted}.
        // If the node was already present, it has already been refined
        // during this pass.
        // budgets don't matter here because we have budget-independent subgraphs
        // iterative budget refinement also counts and does the refinement in full
        if (!visited.insert(node_ptr).second) {
            return;
        }

        const int budget = G->budget;

        if (depth <= 0 || budget < 2 * gamma) {
            return;
        }

        Packed L(n_words), R(n_words);

        const int8_t k_here =
            (proxy_style == 2 &&
            depth > 0 &&
            depth < (int)k_at_depth.size())
                ? k_at_depth[depth - 1]
                : lookahead_init;

        const int anytime_lk1_feat =
            lookup_anytime_lickety_first_split_(mask, depth, pk);

        const bool anytime_lk1_feat_missing_from_active =
            anytime_lk1_feat >= 0 &&
            !feature_in_sorted_vector_(B_active, anytime_lk1_feat);

        const std::size_t split_count_before = G->splits.size();


        // post-order: first refine children of the splits that already existed.
        for (std::size_t idx = 0; idx < split_count_before; ++idx) {
            SplitNode& s = G->splits[idx];

            if (!s.left && !s.right) {
                continue;
            }

            split_threshold_bits_(mask, s.feature, L, R);

            if (!L.any() || !R.any()) {
                continue;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                s.feature,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                s.feature,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            if (s.left) {
                RefineGraphDfs(
                    s.left,
                    L,
                    depth - 1,
                    *pkLp,
                    *cpathLp,
                    B_active,
                    visited
                );
            }

            if (s.right) {
                RefineGraphDfs(
                    s.right,
                    R,
                    depth - 1,
                    *pkRp,
                    *cpathRp,
                    B_active,
                    visited
                );
            }

            // for binary features, we need to do one of the things that enumerate_continuous_feature_for_trie_extend_restricted does for continuous
            // we need to resolve thresholds after we refined them with iterative budget refinement, because of what is below
            const bool is_binary_feature =
                s.feature < first_continuous_feature_();

            const bool is_missing_anytime_lk1_feature =
                anytime_lk1_feat_missing_from_active &&
                s.feature == anytime_lk1_feat;

            if (is_binary_feature || is_missing_anytime_lk1_feature) {
                const int lossL = proxy_completion_objective_(
                    L,
                    depth - 1,
                    k_here,
                    *pkLp,
                    *cpathLp
                );

                const int lossR = proxy_completion_objective_(
                    R,
                    depth - 1,
                    k_here,
                    *pkRp,
                    *cpathRp
                );

                auto LR = solve_siblings_extend(
                    s.left,
                    s.right,
                    lossL,
                    lossR,
                    L,
                    R,
                    budget,
                    depth,
                    *pkLp,
                    *pkRp,
                    *cpathLp,
                    *cpathRp,
                    &B_active,
                    true
                );

                if (
                    !children_have_feasible_pair_(
                        LR.first,
                        LR.second,
                        budget
                    )
                ) {
                    if (resource_limit_reached_()) {
                        return;
                    }
                } else {
                    s.left = LR.first;
                    s.right = LR.second;

                    const int min_sum =
                        s.left->min_objective +
                        s.right->min_objective;

                    if (min_sum < G->min_objective) {
                        G->min_objective = min_sum;
                    }
                }

                
            }
        }

        // now refine this OR node, using children that have already been updated.
        // this is important because we consider existing thresholds in enumerate_continuous_feature_for_trie_extend_restricted
        // and do iterative budget refinement again given that their minimum objectives could have improved with what we just did
        std::unordered_set<int> already_split = local_split_features_(G);

        ContinuousPath pi_cur = materialize_continuous_path_(cpath);

        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start_idx = continuous_starts[(std::size_t)cont_pos];

            const int raw_end_idx =
                (cont_pos + 1 < (int)continuous_starts.size())
                    ? continuous_starts[(std::size_t)(cont_pos + 1)]
                    : n_features;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start_idx,
                    raw_end_idx,
                    pi_cur
                );

            if (start_idx >= end_idx) {
                continue;
            }

            enumerate_continuous_feature_for_trie_extend_restricted(
                G,
                mask,
                depth,
                budget,
                pk,
                pi_cur,
                k_here,
                start_idx,
                end_idx,
                L,
                R,
                already_split,
                &B_active
            );
        }
    }

    // distance between thresholds in active sample distance (i.e., tied to a subproblem)
    inline int bit_distance_between_thresholds_(
        const Packed& mask,
        int feat_a,
        int feat_b
    ) const {
        // number of active samples whose side changes between threshold columns.
        int s = 0;

        const Packed& A = X_bits[(size_t)feat_a];
        const Packed& B = X_bits[(size_t)feat_b];

        return popcount_xor_and_words(
            mask.w.data(),
            A.w.data(),
            B.w.data(),
            n_words
        );
    }

    // we may not need the exact distance, we can terminate early if it is too far away. this is documented in the appendix.
    inline bool bit_distance_at_least_(
        const Packed& mask,
        int feat_a,
        int feat_b,
        int delta
    ) const {
        // need at least 0 moved samples is always true.
        if (delta <= 0) return true;

        // same threshold has distance 0.
        if (feat_a == feat_b) return false;

        int s = 0;

        const Packed& A = X_bits[(size_t)feat_a];
        const Packed& B = X_bits[(size_t)feat_b];

        for (int t = 0; t < n_words; ++t) {
            const uint64_t diff =
                mask.w[(size_t)t] &
                (A.w[(size_t)t] ^ B.w[(size_t)t]);

            s += popcnt64(diff);

            // early exit: tightening only needs to know whether
            // we have moved at least delta active samples.
            if (s >= delta) return true;
        }

        return false;
    }

    // partitioning subproblem into left and right using feat split
    inline void split_threshold_bits_(
        const Packed& mask,
        int feat,
        Packed& L,
        Packed& R
    ) const {
        const Packed& Xf = X_bits[(size_t)feat];

        for (int t = 0; t < n_words; ++t) {
            const uint64_t mw = mask.w[(size_t)t];
            const uint64_t lw = mw & Xf.w[(size_t)t];

            L.w[(size_t)t] = lw;
            R.w[(size_t)t] = mw & ~Xf.w[(size_t)t];
        }

        L.w[(size_t)(n_words - 1)] &= tail_mask;
        R.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    // map is red black tree, so both of these are log
    // returns the largest evaluated threshold index strictly less than i
    // used in some pruning discussed in the proxy section of the appendix
    inline int predecessor_eval_(
        const std::map<int, std::pair<int,int>>& evaluated,
        int i
    ) const {
        auto it = evaluated.lower_bound(i);
        if (it == evaluated.begin()) return -1;
        --it;
        return it->first;
    }

    // smallest strictly larger
    inline int successor_eval_(
        const std::map<int, std::pair<int,int>>& evaluated,
        int j
    ) const {
        auto it = evaluated.upper_bound(j);
        if (it == evaluated.end()) return -1;
        return it->first;
    }

    inline int eval_left_(
        const std::map<int, std::pair<int,int>>& evaluated,
        int feat
    ) const {
        return evaluated.at(feat).first;
    }

    inline int eval_right_(
        const std::map<int, std::pair<int,int>>& evaluated,
        int feat
    ) const {
        return evaluated.at(feat).second;
    }

    // if at budget, that is fine
    inline bool violates_rashomon_bound_(int z, int budget) const {
        return z > budget;
    }

    // need to improve by gap to be within budget
    inline int rashomon_gap_(int z, int budget) const {
        return z - budget;
    }

    // we balance probing and binary search as discussed in the additional methods section of the appendix
    int tighten_lower_bound_bitvector_(
        const Packed& mask,
        int end_idx,
        int u,
        int i,
        int delta
    ) const {
        // return min k >= i such that distance(u,k) >= delta.
        // search interval is [i, end_idx).
        // return end_idx if no such k exists.

        if (delta <= 0) return i;
        if (i >= end_idx) return end_idx;

        // Local probe: nearby thresholds often satisfy the movement condition,
        // and bit_distance_at_least_ can early-exit.
        const int probe_end = std::min(end_idx, i + 2);
        for (int k = i; k < probe_end; ++k) {
            if (bit_distance_at_least_(mask, u, k, delta)) {
                return k;
            }
        }

        // If neither i nor i+1 worked, binary search the remaining suffix.
        int lo = probe_end;
        int hi = end_idx - 1;
        int ans = end_idx;

        while (lo <= hi) {
            const int mid = lo + ((hi - lo) >> 1);

            if (bit_distance_at_least_(mask, u, mid, delta)) {
                ans = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }

        return ans;
    }

    int tighten_upper_bound_bitvector_(
        const Packed& mask,
        int start_idx,
        int v,
        int j,
        int delta
    ) const {
        // return max k <= j such that distance(k,v) >= delta.
        // search interval is [start_idx, j].
        // return start_idx - 1 if no such k exists.

        if (delta <= 0) return j;
        if (j < start_idx) return start_idx - 1;

        // Local probe: nearby thresholds often satisfy the movement condition.
        const int probe_start = std::max(start_idx, j - 1);
        for (int k = j; k >= probe_start; --k) {
            if (bit_distance_at_least_(mask, k, v, delta)) {
                return k;
            }
        }

        // If neither j nor j-1 worked, binary search the remaining prefix.
        int lo = start_idx;
        int hi = probe_start - 1;
        int ans = start_idx - 1;

        while (lo <= hi) {
            const int mid = lo + ((hi - lo) >> 1);

            if (bit_distance_at_least_(mask, mid, v, delta)) {
                ans = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        return ans;
    }

    // start_idx and end_idx are the fixed threshold boundaries of the continuous feature.
    // i and j are the current threshold indices for the interval being pruned.
    std::pair<int,int> shrink_interval_with_bound_bitvector_(
        const std::map<int, std::pair<int,int>>& evaluated,
        const Packed& mask,
        int start_idx,
        int end_idx,
        int i,
        int j,
        int M_L,
        int M_R,
        int budget
    ) const {
        // we will leverage the closest thing we evaluated before, and the closest thing we evaluated above.
        const int u = predecessor_eval_(evaluated, i);
        const int v = successor_eval_(evaluated, j);

        // if the closest evaluated threshold to the left contributes a left side
        // and the closest evaluated threshold to the right contributes a right side
        // whose sum already violates the budget, prune the whole interval, because we haven't even accounted for the points in the middle.
        if (u >= 0 && v >= 0) {
            const int L_u = eval_left_(evaluated, u);
            const int R_v = eval_right_(evaluated, v);

            if (violates_rashomon_bound_(L_u + R_v, budget)) {
                return {1, 0};
            }
        }

        // if predecessor itself violates, move the lower endpoint rightward
        // until enough active samples have changed sides.
        if (u >= 0) {
            const int L_u = eval_left_(evaluated, u);
            const int R_u = eval_right_(evaluated, u);
            const int P_u = L_u + R_u;

            if (violates_rashomon_bound_(P_u, budget)) {
                const int delta = rashomon_gap_(P_u, budget);
                // given a failed predecessor threshold u, and a current interval starting at i, 
                // find the first threshold at or after i that is at least delta samples away from u
                const int new_i = tighten_lower_bound_bitvector_(
                    mask,
                    end_idx,
                    u,
                    i,
                    delta
                );

                i = std::max(i, new_i);
            }
        }

        // if successor itself violates, move the upper endpoint leftward
        // until enough active samples have changed sides.
        if (v >= 0) {
            const int L_v = eval_left_(evaluated, v);
            const int R_v = eval_right_(evaluated, v);
            const int P_v = L_v + R_v;

            if (violates_rashomon_bound_(P_v, budget)) {
                const int delta = rashomon_gap_(P_v, budget);

                const int new_j = tighten_upper_bound_bitvector_(
                    mask,
                    start_idx,
                    v,
                    j,
                    delta
                );

                j = std::min(j, new_j);
            }
        }

        i = std::max(i, M_L);
        j = std::min(j, M_R);

        return {i, j};
    }

    inline int threshold_distance_bitvector_(
        const Packed& mask,
        int a,
        int b
    ) const {
        if (a == b) return 0;

        return popcount_xor_and_words(
            mask.w.data(),
            X_bits[(size_t)a].w.data(),
            X_bits[(size_t)b].w.data(),
            n_words
        );
    }

    // this method is essentially EnumContFeature without InitAndPrune. it is called when we are not extending the graph, so we don't loop over all proxy completions.
    void enumerate_continuous_feature_for_trie(
        shared_ptr<TreeTrieNode>& node,
        const Packed& mask,
        int8_t depth,
        int budget,
        const PathKey& pk,
        ContinuousPath& cpath,
        int8_t k_here,
        int start_idx,
        int end_idx,
        Packed& L,
        Packed& R
    ){
        // continuous threshold group is [start_idx, end_idx).
        // threshold columns are already binarized and monotone ordered.
        if (start_idx >= end_idx) return;

        // evaluated[feat] = {lossL, lossR}
        //
        // feat is the absolute threshold-column feature index.
        // this scratch table includes both feasible and failed evaluated thresholds.
        // successful splits are stored persistently by node->add_split(feat, ...).
        // std::map<int, std::pair<int,int>> evaluated;

        // ContinuousPath q_cpath = materialize_continuous_path_(cpath);
        if ((int)cpath.size() != (int)continuous_starts.size()) {
            initialize_continuous_path_(cpath);
        }
        std::deque<std::pair<int,int>> Q;
        Q.push_back({start_idx, end_idx - 1});

        int M_L = start_idx;
        int M_R = end_idx - 1;

        while (!Q.empty()) {
            auto [i, j] = Q.front();
            Q.pop_front();

            // auto shrunk = shrink_interval_with_bound_bitvector_(
            //     evaluated,
            //     mask,
            //     start_idx,
            //     end_idx,
            //     i,
            //     j,
            //     M_L,
            //     M_R,
            //     budget
            // );

            i = std::max(i, M_L);
            j = std::min(j, M_R);

            if (i > j) continue;

            const int feat = i + ((j - i) >> 1); // midpoint of thresholds, recall we aren't deduplicating

            split_threshold_bits_(mask, feat, L, R);

            const bool left_empty = !L.any();
            const bool right_empty = !R.any();

            if (left_empty || right_empty) {
                if (left_empty && !right_empty) {
                    constrain_continuous_false_branch_(cpath, feat);
                    // threshold too low: all thresholds <= feat also have empty left.
                    // only higher thresholds can become valid.
                    if (feat + 1 <= j) {
                        Q.push_back({feat + 1, j});
                    }
                } else if (!left_empty && right_empty) {
                    constrain_continuous_true_branch_(cpath, feat);
                    // threshold too high: all thresholds >= feat also have empty right.
                    // only lower thresholds can become valid.
                    if (i <= feat - 1) {
                        Q.push_back({i, feat - 1});
                    }
                }
                // if both are empty, mask itself is empty, which should not happen here.
                continue;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                feat,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                feat,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            int lossL;
            int lossR;

            if (lookahead_init < 0) {
                lossL = leaf_objective(L);
            } else if (lookahead_init == 0) {
                lossL = greedy_proxy_objective_(L, depth - 1, *pkLp, *cpathLp);
            } else {
                if (proxy_style == 4) {
                    lossL = split_algorithm(L, depth - 1, k_here, *pkLp);
                } else {
                    lossL = lickety_proxy_objective_(L, depth - 1, k_here, *pkLp, *cpathLp);
                }
            }

            if (lookahead_init < 0) {
                lossR = leaf_objective(R);
            } else if (lookahead_init == 0) {
                lossR = greedy_proxy_objective_(R, depth - 1, *pkRp, *cpathRp);
            } else {
                if (proxy_style == 4) {
                    lossR = split_algorithm(R, depth - 1, k_here, *pkRp);
                } else {
                    lossR = lickety_proxy_objective_(R, depth - 1, k_here, *pkRp, *cpathRp);
                }
            }

            // evaluated[feat] = {lossL, lossR};

            const int total_proxy = lossL + lossR;

            bool feasible;
            if (!rule_list_mode) {
                feasible = (total_proxy <= budget);
            } else {
                feasible = !(lossL > budget - gamma && lossR > budget - gamma);
            }

            if (feasible) {
                std::pair<
                    std::shared_ptr<TreeTrieNode>,
                    std::shared_ptr<TreeTrieNode>
                > LR;

                 if (rule_list_mode || use_multipass) {
                    LR = solve_siblings_extend(
                        nullptr,
                        nullptr,
                        lossL,
                        lossR,
                        L,
                        R,
                        budget,
                        depth,
                        *pkLp,
                        *pkRp,
                        *cpathLp,
                        *cpathRp
                    );
                } else {
                    LR = symmetric_single_pass(
                        lossL,
                        lossR,
                        L,
                        R,
                        budget,
                        depth,
                        *pkLp,
                        *pkRp,
                        *cpathLp,
                        *cpathRp
                    );
                }

                if (
                    !children_have_feasible_pair_(
                        LR.first,
                        LR.second,
                        budget
                    )
                ) {
                    if (resource_limit_reached_()) {
                        return;
                    }
                } else {
                    // persistent successful split storage.
                    // feat is the actual threshold-column index.
                    node->add_split(feat, LR.first, LR.second);
                }

                if (i <= feat - 1) {
                    Q.push_back({i, feat - 1});
                }

                if (feat + 1 <= j) {
                    Q.push_back({feat + 1, j});
                }

                continue; // we enumerate nearby thresholds because they are high quality too
            }

            // evaluated[feat] = {lossL, lossR};

            // failed split pruning.
            // if left is pure/zero-error leaf objective, farther-left thresholds
            // cannot improve the left side enough, so move M_L right.
            if (lossL == gamma) {
                M_L = std::max(M_L, feat + 1);
            }

            // if right is pure/zero-error leaf objective, farther-right thresholds
            // cannot improve the right side enough, so move M_R left.
            if (lossR == gamma) {
                M_R = std::min(M_R, feat - 1);
            }

            const int delta = total_proxy - budget;

            if (delta <= 0) {
                // this is only possible in rule-list mode, where infeasible means
                // both sides individually exceed budget - gamma, not necessarily
                // that lossL + lossR > budget.
                // this is not part of our contribution but an attempt to preserve behavior from existing work
                if (i <= feat - 1) {
                    Q.push_back({i, feat - 1});
                }

                if (feat + 1 <= j) {
                    Q.push_back({feat + 1, j});
                }

                continue;
            }

            const int a = tighten_upper_bound_bitvector_(
                mask,
                start_idx,
                feat,
                feat - 1,
                delta
            );

            const int b = tighten_lower_bound_bitvector_(
                mask,
                end_idx,
                feat,
                feat + 1,
                delta
            );

            if (i <= a) {
                Q.push_back({i, a});
            }

            if (b <= j) {
                Q.push_back({b, j});
            }
        }
    }

    // this is the iterative budget refinement covered in the appendix. the bulk of it is not novel.
    // returns left and right treetrienode. the left and right mask are constants, even as you recurse on construct_trie
    pair<shared_ptr<TreeTrieNode>, shared_ptr<TreeTrieNode>>
    solve_siblings(int loss_l, int loss_r,
               const Packed& Lmask, const Packed& Rmask,
               int budget, int8_t depth,
               const PathKey& pkL, const PathKey& pkR,
               const ContinuousPath& cpathL, const ContinuousPath& cpathR) {
        int left_budget  = budget - loss_r;
        shared_ptr<TreeTrieNode> left_node =
            (left_budget >= 0) ? construct_trie(Lmask, depth - 1, left_budget, pkL, cpathL)
                               : nullptr; // handles some potential issues with non-injective keys
        int min_left = (left_node ? left_node->min_objective : numeric_limits<int>::max());

        int right_budget = (min_left == numeric_limits<int>::max()) ? -1 : (budget - min_left);
        shared_ptr<TreeTrieNode> right_node =
            (right_budget >= 0) ? construct_trie(Rmask, depth - 1, right_budget, pkR, cpathR)
                                : nullptr;
        int min_right = (right_node ? right_node->min_objective : numeric_limits<int>::max());

        while (true) {
            bool improved = false;

            int new_left_budget = (min_right == numeric_limits<int>::max()) ? -1 : (budget - min_right);
            if (new_left_budget > left_budget) {
                left_budget = new_left_budget;
                if (left_budget >= 0) {
                    left_node = construct_trie(Lmask, depth - 1, left_budget, pkL, cpathL);
                    int new_min_left = left_node->min_objective;
                    if (new_min_left < min_left) min_left = new_min_left;
                }
            }

            int new_right_budget = (min_left == numeric_limits<int>::max()) ? -1 : (budget - min_left);
            if (new_right_budget > right_budget) {
                right_budget = new_right_budget;
                if (right_budget >= 0) {
                    right_node = construct_trie(Rmask, depth - 1, right_budget, pkR, cpathR);
                    int new_min_right = right_node->min_objective;
                    if (new_min_right < min_right) { min_right = new_min_right; improved = true; }
                }
            }

            if (!improved) break;
        }

        return {left_node, right_node};
    }

    // this is solely for ableation study purposes - if practical we would subtract minobjective from the other side
    std::pair<std::shared_ptr<TreeTrieNode>, std::shared_ptr<TreeTrieNode>>
    symmetric_single_pass(int loss_l, int loss_r,
                 const Packed& Lmask, const Packed& Rmask,
                 int budget, int8_t depth,
                 const PathKey& pkL, const PathKey& pkR,
                 const ContinuousPath& cpathL, const ContinuousPath& cpathR) {
        int left_budget  = budget - loss_r;
        int right_budget = budget - loss_l;

        std::shared_ptr<TreeTrieNode> left_node  = nullptr;
        std::shared_ptr<TreeTrieNode> right_node = nullptr;

        if (left_budget >= 0) { // robustness incase we change pruning
            left_node = construct_trie(Lmask, depth - 1, left_budget, pkL, cpathL);
        }
        if (right_budget >= 0) {
            right_node = construct_trie(Rmask, depth - 1, right_budget, pkR, cpathR);
        }

        return {left_node, right_node};
    
    }

    // what splits are known to be in the rashomon set at this subproblem?
    static std::unordered_set<int> local_split_features_(
        const std::shared_ptr<TreeTrieNode>& node
    ) {
        std::unordered_set<int> seen;
        if (!node) return seen;

        seen.reserve(node->splits.size() * 2 + 8);

        for (const auto& s : node->splits) {
            seen.insert(s.feature);
        }

        return seen;
    }

    // the more efficient version of solve siblings that can extend directly to bigger budgets
    pair<shared_ptr<TreeTrieNode>, shared_ptr<TreeTrieNode>>
    solve_siblings_extend(
        shared_ptr<TreeTrieNode> left_node,
        shared_ptr<TreeTrieNode> right_node,
        int loss_l,
        int loss_r,
        const Packed& Lmask,
        const Packed& Rmask,
        int budget,
        int8_t depth,
        const PathKey& pkL,
        const PathKey& pkR,
        const ContinuousPath& cpathL,
        const ContinuousPath& cpathR,
        const std::vector<int>* active_features = nullptr,
        bool launched_by_anytime = false
    ) {
        auto solve_or_extend = [&](
            shared_ptr<TreeTrieNode> node,
            const Packed& mask,
            int node_budget,
            const PathKey& pk,
            const ContinuousPath& cpath
        ) -> shared_ptr<TreeTrieNode> {
            if (node_budget < 0) {
                return nullptr;
            }

            if (node) {
                return construct_trie_extend(
                    node,
                    mask,
                    depth - 1,
                    node_budget,
                    pk,
                    cpath,
                    active_features,
                    launched_by_anytime
                );
            }

            return construct_trie(
                mask,
                depth - 1,
                node_budget,
                pk,
                cpath,
                active_features
            );
        };

        // we need to determine which side to solve first, due to caveats with adding thresholds. the appendix discusses this.
        auto solve_ordered = [&](
            shared_ptr<TreeTrieNode>& first_node,
            shared_ptr<TreeTrieNode>& second_node,
            const Packed& first_mask,
            const Packed& second_mask,
            int first_loss,
            int second_loss,
            const PathKey& first_pk,
            const PathKey& second_pk,
            const ContinuousPath& first_cpath,
            const ContinuousPath& second_cpath
        ) {
            int first_budget = first_node
                ? first_node->budget
                : std::numeric_limits<int>::min();

            int second_budget = second_node
                ? second_node->budget
                : std::numeric_limits<int>::min();

            int new_first_budget = budget - second_loss;

            while (new_first_budget > first_budget) {
                first_budget = new_first_budget;

                first_node = solve_or_extend(
                    first_node,
                    first_mask,
                    first_budget,
                    first_pk,
                    first_cpath
                );

                if (!first_node) {
                    return;
                }

                const int first_cert =
                    std::min(first_loss, first_node->min_objective);

                const int new_second_budget =
                    budget - first_cert;

                if (new_second_budget > second_budget) {
                    second_budget = new_second_budget;

                    second_node = solve_or_extend(
                        second_node,
                        second_mask,
                        second_budget,
                        second_pk,
                        second_cpath
                    );

                    if (!second_node) {
                        return;
                    }

                    const int second_cert =
                        std::min(second_loss, second_node->min_objective);

                    new_first_budget =
                        budget - second_cert;
                } else {
                    break;
                }
            }
        };

        const bool both_empty = (!left_node && !right_node);
        const bool both_nonempty = (left_node && right_node);

        if (both_empty) {
            solve_ordered(
                left_node,
                right_node,
                Lmask,
                Rmask,
                loss_l,
                loss_r,
                pkL,
                pkR,
                cpathL,
                cpathR
            );

            return {left_node, right_node};
        }

        if (!both_nonempty) {
            return {nullptr, nullptr};
        }

        const int left_budget = left_node->budget;
        const int right_budget = right_node->budget;

        const int best_left_estimate =
            std::min(loss_l, left_node->min_objective);

        const int best_right_estimate =
            std::min(loss_r, right_node->min_objective);

        const int new_left_budget =
            budget - best_right_estimate;

        const int new_right_budget =
            budget - best_left_estimate;

        if (new_left_budget <= left_budget &&
            new_right_budget <= right_budget) {
            return {left_node, right_node};
        }

        if (new_left_budget > left_budget) {
            solve_ordered(
                left_node,
                right_node,
                Lmask,
                Rmask,
                best_left_estimate,
                best_right_estimate,
                pkL,
                pkR,
                cpathL,
                cpathR
            );
        } else {
            solve_ordered(
                right_node,
                left_node,
                Rmask,
                Lmask,
                best_right_estimate,
                best_left_estimate,
                pkR,
                pkL,
                cpathR,
                cpathL
            );
        }

        return {left_node, right_node};
    }

    // the expected implementation of ContinuousRSet. here, we have active_thresholds (active_features) and directly extend a given node (or we can look up a better one if it exists).
    shared_ptr<TreeTrieNode> construct_trie_extend(shared_ptr<TreeTrieNode> node, const Packed& mask, int8_t depth, int budget, const PathKey& pk, const ContinuousPath& cpath = empty_continuous_path(), const std::vector<int>* active_features = nullptr, bool launched_by_anytime = false) {
        // if (!node) {
        //     return construct_trie(mask, depth, budget, pk, cpath,  active_features);
        // }

        const uint64_t k = key_of_subproblem(mask, pk);
        K2 key{k, depth};

        if (trie_cache_enabled) {
            if (auto it = trie_cache.find(key); it != trie_cache.end()) {
                node = it->second;

                if (budget <= node->budget) {
                    return node;
                }
            }
        }

        if (node && !launched_by_anytime && budget <= node->budget) {
            return node;
        }

        if (resource_limit_reached_()) {
            return node_has_solution_(node) // note that if node is null we preserve that, we don't make an empty node. this ensures solve siblings knows if there is a valid solution or not
                ? node
                : nullptr;
        }

        if (!node) { // no node was given and nothing was found in the cache if we were caching so how can i extend it
            // could call construct_trie, lets just make the node
            node = std::make_shared<TreeTrieNode>();
        }

        node->budget = budget;

        // add newly feasible leaves without duplicating existing predictions.
        {
            std::unordered_set<int> existing_leaf_preds;
            existing_leaf_preds.reserve(node->leaves.size() * 2 + 4);

            for (const auto& leaf : node->leaves) {
                existing_leaf_preds.insert(leaf.prediction);
            }

            int n_sub = 0;

            if (num_classes == 2) {
                int pos = 0;
                int bb_wrong = 0;

                count_total_pos_bbwrong_binary(
                    mask,
                    n_sub,
                    pos,
                    bb_wrong
                );

                const int neg = n_sub - pos;

                if (!majority_leaf_only) {
                    const int cost0 = gamma + pos;
                    const int cost1 = gamma + neg;

                    if (
                        cost0 <= budget &&
                        !existing_leaf_preds.count(0)
                    ) {
                        node->add_leaf(0, cost0);
                        existing_leaf_preds.insert(0);
                    }

                    if (
                        cost1 <= budget &&
                        !existing_leaf_preds.count(1)
                    ) {
                        node->add_leaf(1, cost1);
                        existing_leaf_preds.insert(1);
                    }
                } else {
                    const int best_c = (pos >= neg) ? 1 : 0;
                    const int best_cost =
                        gamma + std::min(pos, neg);

                    if (
                        best_cost <= budget &&
                        !existing_leaf_preds.count(best_c)
                    ) {
                        node->add_leaf(best_c, best_cost);
                        existing_leaf_preds.insert(best_c);
                    }
                }

                if (use_deferral && n_sub > 0) {
                    const int defer_cost =
                        gamma
                        + defer_penalty_from_count_(n_sub)
                        + bb_wrong;

                    if (
                        defer_cost <= budget &&
                        !existing_leaf_preds.count(DEFER_PREDICTION)
                    ) {
                        node->add_leaf(
                            DEFER_PREDICTION,
                            defer_cost
                        );
                        existing_leaf_preds.insert(DEFER_PREDICTION);
                    }
                }
            } else {
                n_sub = count_total(mask);

                std::vector<int> cnts;
                count_per_class(mask, cnts);

                if (!majority_leaf_only) {
                    for (int c = 0; c < num_classes; ++c) {
                        const int mis =
                            n_sub - cnts[(size_t)c];

                        const int cost =
                            gamma + mis;

                        if (
                            cost <= budget &&
                            !existing_leaf_preds.count(c)
                        ) {
                            node->add_leaf(c, cost);
                            existing_leaf_preds.insert(c);
                        }
                    }
                } else {
                    int best_c = 0;
                    int best_cnt = cnts[0];

                    for (int c = 1; c < num_classes; ++c) {
                        const int v = cnts[(size_t)c];

                        if (
                            v > best_cnt ||
                            (v == best_cnt && c > best_c)
                        ) {
                            best_cnt = v;
                            best_c = c;
                        }
                    }

                    const int mis =
                        n_sub - best_cnt;

                    const int best_cost =
                        gamma + mis;

                    if (
                        best_cost <= budget &&
                        !existing_leaf_preds.count(best_c)
                    ) {
                        node->add_leaf(best_c, best_cost);
                        existing_leaf_preds.insert(best_c);
                    }
                }

                if (use_deferral && n_sub > 0) {
                    const int bb_wrong =
                        count_bb_wrong(mask);

                    const int defer_cost =
                        gamma
                        + defer_penalty_from_count_(n_sub)
                        + bb_wrong;

                    if (
                        defer_cost <= budget &&
                        !existing_leaf_preds.count(DEFER_PREDICTION)
                    ) {
                        node->add_leaf(
                            DEFER_PREDICTION,
                            defer_cost
                        );
                        existing_leaf_preds.insert(DEFER_PREDICTION);
                    }
                }
            }
        }

        if (depth == 0 || budget < 2 * gamma) {
            if (!node_has_solution_(node)) {
                return nullptr;
            }

            if (trie_cache_enabled) trie_cache.emplace(key, node);
            return node;
        }

        Packed L(n_words), R(n_words);

        ContinuousPath pi_cur = materialize_continuous_path_(cpath);

        const int8_t k_here = (proxy_style == 2 && depth >= 0 && depth < (int)k_at_depth.size())
            ? k_at_depth[depth-1]
            : lookahead_init;

        
        const int first_continuous_feature = continuous_starts.empty() ? n_features : continuous_starts[0];
        auto already_split = local_split_features_(node);

        // continuous groups, already fully binarized
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start_idx = continuous_starts[(size_t)cont_pos];

            const int raw_end_idx = (cont_pos + 1 < (int)continuous_starts.size())
                ? continuous_starts[(size_t)(cont_pos + 1)]
                : n_features;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start_idx,
                    raw_end_idx,
                    pi_cur
                );

            if (start_idx >= end_idx) {
                continue;
            }

            if (active_features) {
                enumerate_continuous_feature_for_trie_extend_restricted(
                    node,
                    mask,
                    depth,
                    budget,
                    pk,
                    pi_cur,
                    k_here,
                    start_idx,
                    end_idx,
                    L,
                    R,
                    already_split,
                    active_features
                );
            } else {
                // this case is currently unreachable because we pass all active thresholds, preserved for ablation purposed
                 enumerate_continuous_feature_for_trie_extend(
                    node,
                    mask,
                    depth,
                    budget,
                    pk,
                    pi_cur,
                    k_here,
                    start_idx,
                    end_idx,
                    L,
                    R,
                    already_split
                );
            }
            if (resource_limit_reached_()) {
                return node_has_solution_(node)
                    ? node
                    : nullptr;
            }
        }

        for (int f = 0; f < first_continuous_feature; ++f) {
            if (already_split.count(f)) {
                // if this happens, we still need to do the extended solve siblings because we have a bigger budget
                auto it = std::find_if(
                    node->splits.begin(),
                    node->splits.end(),
                    [&](const SplitNode& s) {
                        return s.feature == f;
                    }
                );

                if (it == node->splits.end()) {
                    // Should not happen if already_split was built from node->splits.
                    continue;
                }

                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);

                if (!L.any() || !R.any()) {
                    continue;
                }

                const PathKey* pkLp = &empty_pk();
                const PathKey* pkRp = &empty_pk();

                PathKey pkL_local;
                PathKey pkR_local;

                make_child_pks_if_needed_(
                    f,
                    pk,
                    pkLp,
                    pkRp,
                    pkL_local,
                    pkR_local
                );

                const ContinuousPath* cpathLp = &pi_cur;
                const ContinuousPath* cpathRp = &pi_cur;

                ContinuousPath cpathL_local;
                ContinuousPath cpathR_local;

                make_child_continuous_paths_if_needed_(
                    f,
                    pi_cur,
                    cpathLp,
                    cpathRp,
                    cpathL_local,
                    cpathR_local
                );

                const int lossL = proxy_completion_objective_(
                    L,
                    depth - 1,
                    k_here,
                    *pkLp,
                    *cpathLp
                );

                const int lossR = proxy_completion_objective_(
                    R,
                    depth - 1,
                    k_here,
                    *pkRp,
                    *cpathRp
                );

                auto LR = solve_siblings_extend(
                    it->left,
                    it->right,
                    lossL,
                    lossR,
                    L,
                    R,
                    budget,
                    depth,
                    *pkLp,
                    *pkRp,
                    *cpathLp,
                    *cpathRp,
                    active_features
                );

                if (
                    !children_have_feasible_pair_(
                        LR.first,
                        LR.second,
                        budget
                    )
                ) {
                    if (resource_limit_reached_()) {
                        return node_has_solution_(node)
                            ? node
                            : nullptr;
                    }

                    continue;
                }

                it->left = LR.first;
                it->right = LR.second;

                const int min_sum = it->left->min_objective + it->right->min_objective;
                if (min_sum < node->min_objective) {
                    node->min_objective = min_sum;
                }

                continue;
            }

            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);

            if (!L.any() || !R.any()) continue;

            // build child pks (canonical sorted)
            // pk refs default to EMPTY
            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            // only build PKs in LITS_EXACT
            PathKey pkL_local, pkR_local;
            if (key_mode == KeyMode::LITS_EXACT) {
                pkL_local = pk;
                pkR_local = pk;
                pk_insert_sorted(pkL_local, enc_lit(f, 1));
                pk_insert_sorted(pkR_local, enc_lit(f, 0));
                pkLp = &pkL_local;
                pkRp = &pkR_local;
            }

            int lossL, lossR;

            // to evaluate whether lossL+lossR is within budget (for non rule list mode), we can first handle an early pruning case
            if (lookahead_init < 0) {
                lossL = leaf_objective(L);
            } else if (lookahead_init == 0) {
                lossL = greedy_proxy_objective_(L, depth - 1, *pkLp, pi_cur);
            } else {
                if (proxy_style == 4) {
                    lossL = split_algorithm(L, depth - 1, k_here, *pkLp);
                } else {
                    lossL = lickety_proxy_objective_(L, depth - 1, k_here, *pkLp, pi_cur);
                }
            }

            // either L or R would work here, could take larger, but very cheap to just choose one
            if (!rule_list_mode) {
                if (lossL + gamma > budget) continue;
            }

            // now compute R if we need it for more information
            if (lookahead_init < 0) {
                lossR = leaf_objective(R);
            } else if (lookahead_init == 0) {
                lossR = greedy_proxy_objective_(R, depth - 1, *pkRp, pi_cur);
            } else {
                if (proxy_style == 4) {
                    lossR = split_algorithm(R, depth - 1, k_here, *pkRp);
                } else {
                    lossR = lickety_proxy_objective_(R, depth - 1, k_here, *pkRp, pi_cur);
                }
            }

            // standard pruning logic in paper
            if (!rule_list_mode) {
                if (lossL + lossR > budget) continue; // approximation decision tree rashomon set
            } else {
                if (lossL > budget - gamma && lossR > budget - gamma) continue; // exact rule list rashomon set
            }

            
            // LR has two entries: first and second.
            // these are null because it is a new split
            auto LR = solve_siblings_extend(
                nullptr,
                nullptr,
                lossL,
                lossR,
                L,
                R,
                budget,
                depth,
                *pkLp,
                *pkRp,
                pi_cur,
                pi_cur,
                active_features
            );

            // the left and right TreeTrieNode (OR nodes) to be added to the AND/OR graph being built
            //if (!LR.first || !LR.second) continue; // safeguard, especially needed if we allow non-injective keys
            // we aren't going to return a flag that says if we ran out of resources or not, but we should return null if we ran out of resources
            // so if we get null, we just have to check our resources and find out
            if (
                !children_have_feasible_pair_(
                    LR.first,
                    LR.second,
                    budget
                )
            ) {
                if (resource_limit_reached_()) {
                    const bool has_solution =
                        !node->leaves.empty() ||
                        !node->splits.empty();

                    return has_solution ? node : nullptr;
                }

                continue;
            }

            node->add_split(f, LR.first, LR.second); // add split with left and right subtries
        }

        if (!node_has_solution_(node)) {
            return nullptr;
        }


        if (trie_cache_enabled) {
            trie_cache.emplace(key, node);
        }

        return node;
    }

    // this method is not used in the paper experiments but one can make the else branch where this is called trigger to perform an ablation on InitAndPrune
    void enumerate_continuous_feature_for_trie_extend(
        shared_ptr<TreeTrieNode>& node,
        const Packed& mask,
        int8_t depth,
        int budget,
        const PathKey& pk,
        ContinuousPath& cpath,
        int8_t k_here,
        int start_idx,
        int end_idx,
        Packed& L,
        Packed& R,
        std::unordered_set<int>& already_split
    ){
        // continuous threshold group is [start_idx, end_idx).
        // threshold columns are already binarized and monotone ordered.
        if (start_idx >= end_idx) return;

        // evaluated[feat] = {lossL, lossR}
        //
        // feat is the absolute threshold-column feature index.
        // this scratch table includes both feasible and failed evaluated thresholds.
        // successful splits are stored persistently by node->add_split(feat, ...).
        // std::map<int, std::pair<int,int>> evaluated;

        // ContinuousPath q_cpath = materialize_continuous_path_(cpath);
        if ((int)cpath.size() != (int)continuous_starts.size()) {
            initialize_continuous_path_(cpath);
        }

        std::deque<std::pair<int,int>> Q;
        Q.push_back({start_idx, end_idx - 1});

        int M_L = start_idx;
        int M_R = end_idx - 1;

        while (!Q.empty()) {
            auto [i, j] = Q.front();
            Q.pop_front();

            // auto shrunk = shrink_interval_with_bound_bitvector_(
            //     evaluated,
            //     mask,
            //     start_idx,
            //     end_idx,
            //     i,
            //     j,
            //     M_L,
            //     M_R,
            //     budget
            // );

            i = std::max(i, M_L);
            j = std::min(j, M_R);

            if (i > j) continue;

            const int feat = i + ((j - i) >> 1); // midpoint of thresholds, recall we aren't deduplicating

            if (already_split.count(feat)) {

                auto it = std::find_if(
                    node->splits.begin(),
                    node->splits.end(),
                    [&](const SplitNode& s) {
                        return s.feature == feat;
                    }
                );

                if (it != node->splits.end()) {
                    and_bits(mask, X_bits[feat], L);
                    andnot_bits(mask, X_bits[feat], R);

                    if (L.any() && R.any()) {
                        const PathKey* pkLp = &empty_pk();
                        const PathKey* pkRp = &empty_pk();

                        PathKey pkL_local;
                        PathKey pkR_local;

                        make_child_pks_if_needed_(
                            feat,
                            pk,
                            pkLp,
                            pkRp,
                            pkL_local,
                            pkR_local
                        );

                        const ContinuousPath* cpathLp = &cpath;
                        const ContinuousPath* cpathRp = &cpath;

                        ContinuousPath cpathL_local;
                        ContinuousPath cpathR_local;

                        make_child_continuous_paths_if_needed_(
                            feat,
                            cpath,
                            cpathLp,
                            cpathRp,
                            cpathL_local,
                            cpathR_local
                        );

                        const int lossL = it->left
                            ? it->left->min_objective
                            : std::numeric_limits<int>::max();

                        const int lossR = it->right
                            ? it->right->min_objective
                            : std::numeric_limits<int>::max();

                        auto LR = solve_siblings_extend(
                            it->left,
                            it->right,
                            lossL,
                            lossR,
                            L,
                            R,
                            budget,
                            depth,
                            *pkLp,
                            *pkRp,
                            *cpathLp,
                            *cpathRp
                        );
                        
                        if (
                            !children_have_feasible_pair_(
                                LR.first,
                                LR.second,
                                budget
                            )
                        ) {
                            if (resource_limit_reached_()) {
                                return;
                            }

                            continue;
                        }

                        it->left = LR.first;
                        it->right = LR.second;

                        if (it->left && it->right) {
                            const int min_sum =
                                it->left->min_objective + it->right->min_objective;

                            if (min_sum < node->min_objective) {
                                node->min_objective = min_sum;
                            }
                        }

                    }
                }

                if (i <= feat - 1) {
                    Q.push_back({i, feat - 1});
                }

                if (feat + 1 <= j) {
                    Q.push_back({feat + 1, j});
                }

                continue;
            }

            split_threshold_bits_(mask, feat, L, R);

            const bool left_empty = !L.any();
            const bool right_empty = !R.any();

            if (left_empty || right_empty) {
                if (left_empty && !right_empty) {
                    constrain_continuous_false_branch_(cpath, feat);
                    // threshold too low: all thresholds <= feat also have empty left.
                    // only higher thresholds can become valid.
                    if (feat + 1 <= j) {
                        Q.push_back({feat + 1, j});
                    }
                } else if (!left_empty && right_empty) {
                    constrain_continuous_true_branch_(cpath, feat);
                    // threshold too high: all thresholds >= feat also have empty right.
                    // only lower thresholds can become valid.
                    if (i <= feat - 1) {
                        Q.push_back({i, feat - 1});
                    }
                }
                // if both are empty, mask itself is empty, which should not happen here.
                continue;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                feat,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                feat,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            int lossL;
            int lossR;

            if (lookahead_init < 0) {
                lossL = leaf_objective(L);
            } else if (lookahead_init == 0) {
                lossL = greedy_proxy_objective_(L, depth - 1, *pkLp, *cpathLp);
            } else {
                if (proxy_style == 4) {
                    lossL = split_algorithm(L, depth - 1, k_here, *pkLp);
                } else {
                    lossL = lickety_proxy_objective_(L, depth - 1, k_here, *pkLp, *cpathLp);
                }
            }

            if (lookahead_init < 0) {
                lossR = leaf_objective(R);
            } else if (lookahead_init == 0) {
                lossR = greedy_proxy_objective_(R, depth - 1, *pkRp, *cpathRp);
            } else {
                if (proxy_style == 4) {
                    lossR = split_algorithm(R, depth - 1, k_here, *pkRp);
                } else {
                    lossR = lickety_proxy_objective_(R, depth - 1, k_here, *pkRp, *cpathRp);
                }
            }

            // evaluated[feat] = {lossL, lossR};

            const int total_proxy = lossL + lossR;

            bool feasible;
            if (!rule_list_mode) {
                feasible = (total_proxy <= budget);
            } else {
                feasible = !(lossL > budget - gamma && lossR > budget - gamma);
            }

            if (feasible) {
                auto LR = solve_siblings_extend(
                    nullptr,
                    nullptr,
                    lossL,
                    lossR,
                    L,
                    R,
                    budget,
                    depth,
                    *pkLp,
                    *pkRp,
                    *cpathLp,
                    *cpathRp
                );

                if (
                    !children_have_feasible_pair_(
                        LR.first,
                        LR.second,
                        budget
                    )
                ) {
                    if (resource_limit_reached_()) {
                        return;
                    }
                } else {
                    // persistent successful split storage.
                    // feat is the actual threshold-column index.
                    node->add_split(feat, LR.first, LR.second);
                }

                if (i <= feat - 1) {
                    Q.push_back({i, feat - 1});
                }

                if (feat + 1 <= j) {
                    Q.push_back({feat + 1, j});
                }

                continue; // we enumerate nearby thresholds because they are high quality too
            }

            // evaluated[feat] = {lossL, lossR};

            // failed split pruning.
            // if left is pure/zero-error leaf objective, farther-left thresholds
            // cannot improve the left side enough, so move M_L right.
            if (lossL == gamma) {
                M_L = std::max(M_L, feat + 1);
            }

            // if right is pure/zero-error leaf objective, farther-right thresholds
            // cannot improve the right side enough, so move M_R left.
            if (lossR == gamma) {
                M_R = std::min(M_R, feat - 1);
            }

            const int delta = total_proxy - budget;

            if (delta <= 0) {
                // this is only possible in rule-list mode, where infeasible means
                // both sides individually exceed budget - gamma, not necessarily
                // that lossL + lossR > budget.
                if (i <= feat - 1) {
                    Q.push_back({i, feat - 1});
                }

                if (feat + 1 <= j) {
                    Q.push_back({feat + 1, j});
                }

                continue;
            }

            const int a = tighten_upper_bound_bitvector_(
                mask,
                start_idx,
                feat,
                feat - 1,
                delta
            );

            const int b = tighten_lower_bound_bitvector_(
                mask,
                end_idx,
                feat,
                feat + 1,
                delta
            );

            if (i <= a) {
                Q.push_back({i, a});
            }

            if (b <= j) {
                Q.push_back({b, j});
            }
        }
    }


    inline void count_total_pos_binary(const Packed& mask, int& n, int& pos) const {
        const Packed& Ypos = Y_bits[(size_t)1];
        count_total_and_pos_words(mask.w.data(), Ypos.w.data(), n_words, n, pos);
    }

    static inline void count_total_and_pos_words(
        const uint64_t* mask,
        const uint64_t* ypos,
        int n_words,
        int& n,
        int& pos
    ) {
    #if ArborEnum_USE_AVX512_POPCNT
        int i = 0;
        __m512i acc_n = _mm512_setzero_si512();
        __m512i acc_p = _mm512_setzero_si512();

        for (; i + 8 <= n_words; i += 8) {
            __m512i vm = _mm512_loadu_si512((const void*)(mask + i));
            __m512i vy = _mm512_loadu_si512((const void*)(ypos + i));
            __m512i vp = _mm512_and_si512(vm, vy);

            acc_n = _mm512_add_epi64(acc_n, _mm512_popcnt_epi64(vm));
            acc_p = _mm512_add_epi64(acc_p, _mm512_popcnt_epi64(vp));
        }

        alignas(64) uint64_t tmp_n[8];
        alignas(64) uint64_t tmp_p[8];

        _mm512_store_si512((void*)tmp_n, acc_n);
        _mm512_store_si512((void*)tmp_p, acc_p);

        uint64_t tn =
            tmp_n[0] + tmp_n[1] + tmp_n[2] + tmp_n[3] +
            tmp_n[4] + tmp_n[5] + tmp_n[6] + tmp_n[7];

        uint64_t tp =
            tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] +
            tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];

        for (; i < n_words; ++i) {
            const uint64_t mw = mask[i];
            tn += (uint64_t)popcnt64(mw);
            tp += (uint64_t)popcnt64(mw & ypos[i]);
        }

        n = (int)tn;
        pos = (int)tp;
    #else
        int tn = 0;
        int tp = 0;
        for (int i = 0; i < n_words; ++i) {
            const uint64_t mw = mask[i];
            tn += popcnt64(mw);
            tp += popcnt64(mw & ypos[i]);
        }
        n = tn;
        pos = tp;
    #endif
    }

    inline void count_total_pos_bbwrong_binary(
        const Packed& mask,
        int& n,
        int& pos,
        int& bb_wrong
    ) const {
        n = 0;
        pos = 0;
        bb_wrong = 0;

        const Packed& Ypos = Y_bits[(size_t)1];

        if (use_deferral) {
            for (int i = 0; i < n_words; ++i) {
                const uint64_t mw = mask.w[(size_t)i];

                n += popcnt64(mw);
                pos += popcnt64(
                    mw & Ypos.w[(size_t)i]
                );
                bb_wrong += popcnt64(
                    mw & BBwrong.w[(size_t)i]
                );
            }
        } else {
            for (int i = 0; i < n_words; ++i) {
                const uint64_t mw = mask.w[(size_t)i];

                n += popcnt64(mw);
                pos += popcnt64(
                    mw & Ypos.w[(size_t)i]
                );
            }
        }
    }

    struct BestLeafAction {
        int prediction = 0; // 0..C-1 or DEFER_PREDICTION
        int loss = 0;
    };

    BestLeafAction best_leaf_action(
        const Packed& mask
    ) const {
        if (num_classes == 2) {
            int n_sub = 0;
            int pos = 0;
            int bb_wrong = 0;

            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos,
                bb_wrong
            );

            if (n_sub == 0) {
                return BestLeafAction{0, 0};
            }

            const int neg = n_sub - pos;

            BestLeafAction best;

            // class 1 wins ties.
            if (pos >= neg) {
                best.prediction = 1;
                best.loss = gamma + neg;
            } else {
                best.prediction = 0;
                best.loss = gamma + pos;
            }

            if (use_deferral) {
                const int defer_loss =
                    gamma
                    + defer_penalty_from_count_(n_sub)
                    + bb_wrong;

                // preserve the ordinary prediction on ties
                if (defer_loss < best.loss) {
                    best.prediction = DEFER_PREDICTION;
                    best.loss = defer_loss;
                }
            }

            return best;
        }

        const int n_sub = count_total(mask);

        if (n_sub == 0) {
            return BestLeafAction{0, 0};
        }

        int best_pred = 0;
        int best_cnt = -1;

        for (int c = 0; c < num_classes; ++c) {
            const int cnt = count_class(mask, c);

            if (
                cnt > best_cnt ||
                (cnt == best_cnt && c > best_pred)
            ) {
                best_cnt = cnt;
                best_pred = c;
            }
        }

        BestLeafAction best;
        best.prediction = best_pred;
        best.loss = gamma + (n_sub - best_cnt);

        if (use_deferral) {
            const int bb_wrong = count_bb_wrong(mask);

            const int defer_loss =
                gamma
                + defer_penalty_from_count_(n_sub)
                + bb_wrong;

            if (defer_loss < best.loss) {
                best.prediction = DEFER_PREDICTION;
                best.loss = defer_loss;
            }
        }

        return best;
    }

    int leaf_objective(const Packed& mask) const {
        return best_leaf_action(mask).loss;
    }

    inline void make_child_pks_if_needed_(
        int feat,
        const PathKey& pk,
        const PathKey*& pkLp,
        const PathKey*& pkRp,
        PathKey& pkL_local,
        PathKey& pkR_local
    ) const {
        pkLp = &empty_pk();
        pkRp = &empty_pk();
        if (key_mode == KeyMode::LITS_EXACT) {
            pkL_local = pk;
            pkR_local = pk;
            pk_insert_sorted(pkL_local, enc_lit(feat, 1));
            pk_insert_sorted(pkR_local, enc_lit(feat, 0));
            pkLp = &pkL_local;
            pkRp = &pkR_local;
        }
    }

    inline bool is_continuous_threshold_feature_(int feat) const {
        return feat >= first_continuous_feature_() && feat < n_features;
    }

    // apply restrictions based on our non-constant lower and upper bounds, those are stored in cpath in this implementation
    inline std::pair<int,int> tighten_continuous_interval_from_path_(
        int start_idx,
        int end_idx,
        const ContinuousPath& cpath
    ) const {
        if (cpath.empty()) {
            return {start_idx, end_idx};
        }

        auto it = std::lower_bound(
            continuous_starts.begin(),
            continuous_starts.end(),
            start_idx
        );

        if (it == continuous_starts.end() || *it != start_idx) {
            return {start_idx, end_idx};
        }

        const int g = (int)std::distance(continuous_starts.begin(), it);

        if (g < 0 || g >= (int)cpath.size()) {
            return {start_idx, end_idx};
        }

        const ContinuousPathEntry& b = cpath[(std::size_t)g];

        if (b.lo < 0 || b.hi < 0) {
            return {start_idx, end_idx};
        }

        const int lo = std::max(start_idx, b.lo);
        const int hi = std::min(end_idx, b.hi);

        return {lo, hi};
    }

    inline void make_child_continuous_paths_if_needed_(
        int feat,
        const ContinuousPath& cpath,
        const ContinuousPath*& cpathLp,
        const ContinuousPath*& cpathRp,
        ContinuousPath& cpathL_local,
        ContinuousPath& cpathR_local
    ) const {
        cpathLp = &cpath;
        cpathRp = &cpath;

        if (!is_continuous_threshold_feature_(feat)) {
            return;
        }

        cpathL_local = materialize_continuous_path_(cpath);
        cpathR_local = cpathL_local;

        constrain_continuous_true_branch_(cpathL_local, feat);
        constrain_continuous_false_branch_(cpathR_local, feat);

        cpathLp = &cpathL_local;
        cpathRp = &cpathR_local;
    }



    inline int defer_penalty_from_count_(int n_sub) const {
        return static_cast<int>(
            std::llround(
                eta_defer * static_cast<double>(n_sub)
            )
        );
    }

    inline int count_bb_wrong(const Packed& mask) const {
        if (!use_deferral) return 0;
        return popcount_and(mask, BBwrong);
    }

    inline bool bb_wrong_at_(int row) const {
        if (!use_deferral) {
            return false;
        }

        return (
            BBwrong.w[(size_t)(row >> 6)] &
            (1ULL << (row & 63))
        ) != 0ULL;
    }

    // prediction-only objective. Useful only when defer must be excluded.
    inline int predict_leaf_objective_binary_from_counts(
        int n_sub,
        int pos
    ) const {
        if (n_sub == 0) return 0;
        return gamma + std::min(pos, n_sub - pos);
    }

    inline int leaf_objective_binary_from_counts(
        int n_sub,
        int pos
    ) const {
        if (use_deferral && n_sub > 0) {
            throw std::logic_error(
                "Deferral-aware binary leaf objective requires "
                "the subproblem black-box mistake count."
            );
        }

        return leaf_objective_binary_from_counts(
            n_sub,
            pos,
            0
        );
    }

    inline int leaf_objective_binary_from_counts(
        int n_sub,
        int pos,
        int bb_wrong
    ) const {
        if (n_sub == 0) return 0;

        const int predict_loss =
            gamma + std::min(pos, n_sub - pos);

        if (!use_deferral) {
            return predict_loss;
        }

        const int defer_loss =
            gamma
            + defer_penalty_from_count_(n_sub)
            + bb_wrong;

        return std::min(
            predict_loss,
            defer_loss
        );
    }

    // continuous land
    enum class ContinuousEvalMode {
        Lickety,
        Exact
    };

    // called the incumbent solution in the appendix
    struct ContinuousBestSplitResult {
        int best_sum = std::numeric_limits<int>::max();          // feature-selection score
        int best_cached_sum = std::numeric_limits<int>::max();   // objective-only candidate
        int best_feat = -1;
    };

    inline int first_continuous_feature_() const {
        return continuous_starts.empty() ? n_features : continuous_starts[0];
    }

    inline int continuous_group_end_(int cont_pos) const {
        return (cont_pos + 1 < (int)continuous_starts.size())
            ? continuous_starts[(size_t)(cont_pos + 1)]
            : n_features;
    }

    inline int continuous_group_pos_for_threshold_(int feat) const {
        if (!is_continuous_threshold_feature_(feat)) {
            return -1;
        }

        auto it = std::upper_bound(
            continuous_starts.begin(),
            continuous_starts.end(),
            feat
        );

        if (it == continuous_starts.begin()) {
            return -1;
        }

        const int g = (int)std::distance(continuous_starts.begin(), it) - 1;
        const int end = continuous_group_end_(g);

        if (feat < continuous_starts[(std::size_t)g] || feat >= end) {
            return -1;
        }

        return g;
    }

    inline void initialize_continuous_path_(ContinuousPath& cpath) const {
        cpath.resize((std::size_t)continuous_starts.size());

        for (int g = 0; g < (int)continuous_starts.size(); ++g) {
            cpath[(std::size_t)g].lo = continuous_starts[(std::size_t)g];
            cpath[(std::size_t)g].hi = continuous_group_end_(g);
        }
    }

    inline ContinuousPath materialize_continuous_path_(
        const ContinuousPath& cpath
    ) const {
        if ((int)cpath.size() == (int)continuous_starts.size()) {
            return cpath;
        }

        ContinuousPath out;
        initialize_continuous_path_(out);
        return out;
    }

    inline void constrain_continuous_true_branch_(
        ContinuousPath& cpath,
        int feat
    ) const {
        const int g = continuous_group_pos_for_threshold_(feat);
        if (g < 0) return;

        if ((int)cpath.size() != (int)continuous_starts.size()) {
            initialize_continuous_path_(cpath);
        }

        // x <= feat, so thresholds >= feat are all-true on this child.
        cpath[(std::size_t)g].hi =
            std::min(cpath[(std::size_t)g].hi, feat);
    }

    inline void constrain_continuous_false_branch_(
        ContinuousPath& cpath,
        int feat
    ) const {
        const int g = continuous_group_pos_for_threshold_(feat);
        if (g < 0) return;

        if ((int)cpath.size() != (int)continuous_starts.size()) {
            initialize_continuous_path_(cpath);
        }

        // x > feat, so thresholds <= feat are all-false on this child.
        cpath[(std::size_t)g].lo =
            std::max(cpath[(std::size_t)g].lo, feat + 1);
    }

    // best-search mode: equality is not useful, because we only need strict improvement.
    inline bool violates_best_bound_(int z, int best) const {
        return z >= best;
    }

    // if z == best, we still need at least one active sample to move before
    // a neighboring threshold can possibly become strictly better.
    inline int best_gap_(int z, int best) const {
        return std::max(1, z - best);
    }

    std::pair<int,int> shrink_interval_with_best_bound_bitvector_(
        const std::map<int, std::pair<int,int>>& evaluated,
        const Packed& mask,
        int start_idx,
        int end_idx,
        int i,
        int j,
        int M_L,
        int M_R,
        int best
    ) const {
        const int u = predecessor_eval_(evaluated, i);
        const int v = successor_eval_(evaluated, j);

        // if predecessor-left plus successor-right already cannot strictly beat best,
        // then every threshold in the middle is hopeless.
        if (u >= 0 && v >= 0) {
            const int L_u = eval_left_(evaluated, u);
            const int R_v = eval_right_(evaluated, v);

            if (violates_best_bound_(L_u + R_v, best)) {
                return {1, 0};
            }
        }

        // if predecessor total cannot beat best, move lower endpoint right until
        // enough active samples have changed sides to make strict improvement possible.
        if (u >= 0) {
            const int L_u = eval_left_(evaluated, u);
            const int R_u = eval_right_(evaluated, u);
            const int P_u = L_u + R_u;

            if (violates_best_bound_(P_u, best)) {
                const int delta = best_gap_(P_u, best);

                const int new_i = tighten_lower_bound_bitvector_(
                    mask,
                    end_idx,
                    u,
                    i,
                    delta
                );

                i = std::max(i, new_i);
            }
        }

        // if successor total cannot beat best, move upper endpoint left until
        // enough active samples have changed sides to make strict improvement possible.
        if (v >= 0) {
            const int L_v = eval_left_(evaluated, v);
            const int R_v = eval_right_(evaluated, v);
            const int P_v = L_v + R_v;

            if (violates_best_bound_(P_v, best)) {
                const int delta = best_gap_(P_v, best);

                const int new_j = tighten_upper_bound_bitvector_(
                    mask,
                    start_idx,
                    v,
                    j,
                    delta
                );

                j = std::min(j, new_j);
            }
        }

        i = std::max(i, M_L);
        j = std::min(j, M_R);

        return {i, j};
    }

    template <typename EvalFeatureFn>
    void maybe_eval_continuous_greedy_suggested_split_(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk,
        const std::vector<int>& feats,
        EvalFeatureFn&& eval_feature
    ) {
        if (!should_add_continuous_greedy_split_to_binary_lickety_()) {
            return;
        }

        if (feats.empty()) {
            return; // already scanning all features
        }

        const GreedyObjFirstSplit g =
            train_greedy_continuous_with_first_split_(
                mask,
                depth_budget,
                pk,
                empty_continuous_path()
            );

        const int gf = g.first_feat;

        if (gf < 0 || gf >= n_features) {
            return;
        }

        if (std::binary_search(feats.begin(), feats.end(), gf)) {
            return; // already evaluated by binary Lickety
        }

        eval_feature(gf);
    }

    int eval_with_lookahead_continuous(
        const Packed& mask,
        int8_t depth_budget,
        int8_t k,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (depth_budget <= 0) {
            return leaf_objective(mask);
        }

        if (depth_budget == 1) {
            return depthd_exact_proxy_objective_(mask, 1, pk, cpath);
        }

        if (k <= 0) {
            return greedy_proxy_objective_(mask, depth_budget, pk, cpath);
        }

        return generalized_lickety_split_continuous(mask, depth_budget, k, pk, cpath);
    }

    // this is how we choose the best split whether in licketysplit/snip or optimal modes
    // this implements the best split selector in the appendix (proxy algorithms section)
    ContinuousBestSplitResult search_continuous_feature_for_best_split_continuous_(
        const Packed& mask,
        int8_t depth_budget,
        int8_t child_k,
        int8_t cache_lookup_k,
        const PathKey& pk,
        ContinuousPath& cpath,
        int start_idx,
        int end_idx,
        int current_best,
        ContinuousEvalMode eval_mode
    ) {
        // unlike rashomon, if we are just current best at best, can prune. can also update current best
        ContinuousBestSplitResult out;
        out.best_sum = current_best;
        out.best_cached_sum = std::numeric_limits<int>::max();
        out.best_feat = -1;

        auto tightened = tighten_continuous_interval_from_path_(start_idx, end_idx, cpath);

        start_idx = tightened.first;
        end_idx = tightened.second;

        if (start_idx >= end_idx) return out;
        
        std::map<int, std::pair<int,int>> evaluated;

        // ContinuousPath q_cpath = materialize_continuous_path_(cpath);
        if ((int)cpath.size() != (int)continuous_starts.size()) {
            initialize_continuous_path_(cpath);
        }

        std::deque<std::pair<int,int>> Q;
        Q.push_back({start_idx, end_idx - 1});

        int M_L = start_idx;
        int M_R = end_idx - 1;

        Packed L(n_words), R(n_words);

        while (!Q.empty()) {
            auto [i, j] = Q.front();
            Q.pop_front();

            auto shrunk = shrink_interval_with_best_bound_bitvector_(
                evaluated,
                mask,
                start_idx,
                end_idx,
                i,
                j,
                M_L,
                M_R,
                out.best_sum
            );

            i = shrunk.first;
            j = shrunk.second;

            if (i > j) continue;

            const int feat = i + ((j - i) >> 1);

            split_threshold_bits_(mask, feat, L, R);

            const bool left_empty = !L.any();
            const bool right_empty = !R.any();

            if (left_empty || right_empty) {
                if (left_empty && !right_empty) {
                    constrain_continuous_false_branch_(cpath, feat);
                    if (feat + 1 <= j) {
                        Q.push_back({feat + 1, j});
                    }
                } else if (!left_empty && right_empty) {
                    constrain_continuous_true_branch_(cpath, feat);
                    if (i <= feat - 1) {
                        Q.push_back({i, feat - 1});
                    }
                }
                continue;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                feat,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                feat,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            int lossL;
            int lossR;

            if (eval_mode == ContinuousEvalMode::Exact) {
                if (depth_budget == 1) { // note that this is saying depth_budget-1 recursive call would give leaf objective, which is correct, it isn't that depth budget 1 is leaf.
                    lossL = leaf_objective(L);
                    lossR = leaf_objective(R);
                } else {
                    lossL = depthd_exact_solver_cached_continuous(
                        L,
                        depth_budget - 1,
                        *pkLp,
                        *cpathLp
                    );

                    lossR = depthd_exact_solver_cached_continuous(
                        R,
                        depth_budget - 1,
                        *pkRp,
                        *cpathRp
                    );
                }
            } else {
                lossL = eval_with_lookahead_continuous(
                    L,
                    depth_budget - 1,
                    child_k,
                    *pkLp,
                    *cpathLp
                );

                lossR = eval_with_lookahead_continuous(
                    R,
                    depth_budget - 1,
                    child_k,
                    *pkRp,
                    *cpathRp
                );
            }

            const int sum = lossL + lossR;

            evaluated[feat] = {lossL, lossR};

            if (sum < out.best_sum) {
                out.best_sum = sum;
                out.best_feat = feat;
            }

            update_cached_rollout_sum_if_available_(
                out.best_cached_sum,
                L,
                R,
                (int8_t)(depth_budget - 1),
                cache_lookup_k,
                *pkLp,
                *pkRp,
                lossL,
                lossR
            );

            // one-sided pure-side cutoffs.
            // if the left side is already at minimum possible leaf penalty,
            // farther-left thresholds cannot improve the left side.
            if (lossL == gamma) {
                M_L = std::max(M_L, feat + 1);
            }

            // if the right side is already at minimum possible leaf penalty,
            // farther-right thresholds cannot improve the right side.
            if (lossR == gamma) {
                M_R = std::min(M_R, feat - 1);
            }

            // we are bad, we can prune our neighbors
            if (sum >= out.best_sum) {
                const int delta = best_gap_(sum, out.best_sum); // if we are equal, we must move at least one sample

                const int a = tighten_upper_bound_bitvector_(
                    mask,
                    start_idx,
                    feat,
                    feat - 1,
                    delta
                );

                const int b = tighten_lower_bound_bitvector_(
                    mask,
                    end_idx,
                    feat,
                    feat + 1,
                    delta
                );

                if (i <= a) {
                    Q.push_back({i, a});
                }

                if (b <= j) {
                    Q.push_back({b, j});
                }
            } // else { // should never happen
            //     if (i <= feat - 1) {
            //         Q.push_back({i, feat - 1});
            //     }

            //     if (feat + 1 <= j) {
            //         Q.push_back({feat + 1, j});
            //     }
            // }
        }

        return out;
    }

    // this is "licketysnip" with some lookahead. licketysnip with k (lookahead) 1 is what we evalaute in our paper.
    // using k >= depth_budget-1 recovers the optimal proxy that we use in our paper.
    int generalized_lickety_split_continuous(
        const Packed& mask,
        int8_t depth_budget,
        int8_t k,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (depth_budget == 0) {
            return leaf_objective(mask);
        }

        const bool depthd_mode_matches_lickety = !use_restricted_depthd_exact_proxy_();
        const bool not_greedy_opt_binarized = !should_route_continuous_lickety_depth1_to_binary_greedy_();

        if (depth_budget == 1 && not_greedy_opt_binarized) {
            if (depthd_mode_matches_lickety) {
                return depthd_exact_proxy_objective_(mask, 1, pk, cpath);
            }

            return greedy_proxy_objective_(mask, 1, pk, cpath);
        }


        if (k > depth_budget - 1) {
            k = depth_budget - 1;
        }

        if (k == depth_budget - 1 && depthd_mode_matches_lickety && not_greedy_opt_binarized) {
            return depthd_exact_proxy_objective_(mask, depth_budget, pk, cpath);
        }

        uint64_t kmask = 0;
        K2  key2{0, depth_budget};
        KLA keyla{0, depth_budget, k};

        const bool use_kla = use_kla_cache();

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key2.k = kmask;
            keyla.k = kmask;

            if (use_kla) {
                if (auto it = lickety_cache_kla.find(keyla); it != lickety_cache_kla.end()) {
                    return it->second;
                }
            } else {
                if (auto it = lickety_cache_k2.find(key2); it != lickety_cache_k2.end()) {
                    return it->second;
                }
            }
        }

        const int leaf_loss = leaf_objective(mask);

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                if (use_kla) {
                    lickety_cache_kla.emplace(keyla, leaf_loss);
                } else {
                    lickety_cache_k2.emplace(key2, leaf_loss);
                }
            }

            return leaf_loss;
        }

        int best_feat = -1;
        int best_sum = leaf_loss;
        int best_cached_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words), bestL(n_words), bestR(n_words);

        const int8_t child_k = k - 1;

        // const int F = proxy_feat_count_();
        // const int first_cont = std::min(first_continuous_feature_(), F);
        const int F = n_features;
        const int first_cont = first_continuous_feature_();

        ContinuousPath pi_cur = materialize_continuous_path_(cpath);

        // continuous groups.
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start = continuous_starts[(size_t)cont_pos];
            const int raw_end = continuous_group_end_(cont_pos);

            if (raw_start >= F) continue;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start,
                    std::min(raw_end, F),
                    pi_cur
                );

            if (start_idx >= end_idx) continue;

            ContinuousBestSplitResult cres =
                search_continuous_feature_for_best_split_continuous_(
                    mask,
                    depth_budget,
                    child_k,
                    /*cache_lookup_k=*/k,
                    pk,
                    pi_cur,
                    start_idx,
                    end_idx,
                    best_sum,
                    ContinuousEvalMode::Lickety
                );

            if (cres.best_feat >= 0 && cres.best_sum < best_sum) {
                best_sum = cres.best_sum;
                best_feat = cres.best_feat;
            }

            if (cres.best_cached_sum < best_cached_sum) {
                best_cached_sum = cres.best_cached_sum;
            }
        }

        // binary / ordinary feature columns only.
        for (int f = 0; f < first_cont; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);

            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                f,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const int left_loss = eval_with_lookahead_continuous(
                L,
                depth_budget - 1,
                child_k,
                *pkLp,
                pi_cur
            );

            const int right_loss = eval_with_lookahead_continuous(
                R,
                depth_budget - 1,
                child_k,
                *pkRp,
                pi_cur
            );

            const int sum = left_loss + right_loss;

            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
            }

            update_cached_rollout_sum_if_available_(
                best_cached_sum,
                L,
                R,
                (int8_t)(depth_budget - 1),
                /*lookup_k=*/k,
                *pkLp,
                *pkRp,
                left_loss,
                right_loss
            );
        }

        int ans = leaf_loss;

        int8_t k_recurse;

        ans = std::min(ans, best_sum);

        if (best_cached_sum != std::numeric_limits<int>::max()) {
            ans = std::min(ans, best_cached_sum);
        }
        
        // this isn't used in our experiments; it can optionally improve a completion.
        tighten_with_trie_min_if_available_(ans, kmask, depth_budget);

        if (proxy_style == 0) {
            // style 0: constant k.
            k_recurse = k;
        } else if (proxy_style == 3) {
            // SPLIT without postprocessing: tree is fully determined by the chosen split.
            ans = std::min(ans, best_sum);

            if (proxy_caching_enabled) {
                if (use_kla) {
                    lickety_cache_kla.emplace(keyla, ans);
                } else {
                    lickety_cache_k2.emplace(key2, ans);
                }
            }

            return ans;
        } else {
            // styles 1/2: recursively cycle k, k-1, ..., 1, k.
            k_recurse = (child_k == 0) ? lookahead_init : child_k;
        }

        if (best_feat >= 0) { // now partition for the best threshold
            split_threshold_bits_(mask, best_feat, bestL, bestR);

            const int8_t next_depth = depth_budget - 1;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                best_feat,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &pi_cur;
            const ContinuousPath* cpathRp = &pi_cur;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                best_feat,
                pi_cur,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            const int left_loss = generalized_lickety_split_continuous(
                bestL,
                next_depth,
                k_recurse,
                *pkLp,
                *cpathLp
            );

            const int right_loss = generalized_lickety_split_continuous(
                bestR,
                next_depth,
                k_recurse,
                *pkRp,
                *cpathRp
            );

            ans = std::min(ans, left_loss + right_loss);
            ans = std::min(ans, best_sum);
        }

        if (anytime_continuous_lickety_k1_first_split_enabled_() && k == 1 && best_feat >= 0) {
            anytime_lickety_first_split_cache[key2] = best_feat;
        }

        if (proxy_caching_enabled) {
            if (use_kla) {
                lickety_cache_kla.emplace(keyla, ans);
            } else {
                lickety_cache_k2.emplace(key2, ans);
            }
        }

        return ans;
    }

    // getting the sorted lists of indices that we need for this other greedy mode
    NumericalGreedyState make_numerical_state_for_mask_(
        const Packed& mask
    ) const {
        if (numerical_X_cols_for_greedy.size() != continuous_starts.size() ||
            numerical_global_sorted_idx.size() != continuous_starts.size() ||
            numerical_unique_values_for_greedy.size() != continuous_starts.size()) {
            throw std::logic_error(
                "Numerical greedy representation is not aligned with continuous_starts."
            );
        }

        NumericalGreedyState state;
        state.sorted_idx_by_num.resize(numerical_global_sorted_idx.size());

        for (std::size_t g = 0; g < numerical_global_sorted_idx.size(); ++g) {
            const auto& global_order = numerical_global_sorted_idx[g];
            auto& active_order = state.sorted_idx_by_num[g];

            active_order.clear();
            active_order.reserve(global_order.size());

            for (int row : global_order) {
                if (mask_has_row_(mask, row)) {
                    active_order.push_back(row);
                }
            }
        }

        return state;
    }

    // just a wrapper for the optimal decision tree here. it calls the existing best-split selector with the correct lookahead
    int depthd_exact_solver_cached_continuous(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (depth_budget <= 0) {
            return leaf_objective(mask);
        }

        const int8_t DEPTH = depth_budget;
        const int8_t KTAG = depth_budget - 1;

        uint64_t kmask = 0;
        int cached = 0;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);

            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) {
                return cached;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos,
                bb_wrong
            );

            leaf_loss =
                leaf_objective_binary_from_counts(
                    n_sub,
                    pos,
                    bb_wrong
                );
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(
                    kmask,
                    DEPTH,
                    KTAG,
                    leaf_loss,
                    /*allow_cache=*/cache_cheap_subproblems
                );
            }

            return leaf_loss;
        }

        int best_sum = leaf_loss;
        int best_cached_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);

        const int F = n_features;
        const int first_cont = first_continuous_feature_();

        ContinuousPath pi_cur = materialize_continuous_path_(cpath);


        // continuous groups
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start = continuous_starts[(size_t)cont_pos];
            const int raw_end = continuous_group_end_(cont_pos);

            if (raw_start >= F) continue;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start,
                    std::min(raw_end, F),
                    pi_cur
                );

            if (start_idx >= end_idx) continue;

            ContinuousBestSplitResult cres =
                search_continuous_feature_for_best_split_continuous_(
                    mask,
                    depth_budget,
                    /*child_k=*/depth_budget - 2,
                    /*cache_lookup_k=*/depth_budget - 2,
                    pk,
                    pi_cur,
                    start_idx,
                    end_idx,
                    best_sum,
                    ContinuousEvalMode::Exact
                );

            if (cres.best_feat >= 0 && cres.best_sum < best_sum) {
                best_sum = cres.best_sum;
            }

            if (cres.best_cached_sum < best_cached_sum) {
                best_cached_sum = cres.best_cached_sum;
            }
        }

        // binary / ordinary features.
        for (int f = 0; f < first_cont; ++f) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[f], L, R, left_n);

                if (left_n == 0 || left_n == n_sub) {
                    continue;
                }
            } else {
                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);

                if (!L.any() || !R.any()) {
                    continue;
                }
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                f,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &pi_cur;
            const ContinuousPath* cpathRp = &pi_cur;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                f,
                pi_cur,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            int left_best;
            int right_best;

            if (depth_budget == 1) {
                left_best = leaf_objective(L);
                right_best = leaf_objective(R);
            } else {
                left_best = depthd_exact_solver_cached_continuous(
                    L,
                    depth_budget - 1,
                    *pkLp,
                    *cpathLp
                );

                right_best = depthd_exact_solver_cached_continuous(
                    R,
                    depth_budget - 1,
                    *pkRp,
                    *cpathRp
                );
            }

            const int sum = left_best + right_best;

            if (sum < best_sum) {
                best_sum = sum;
            }

            update_cached_rollout_sum_if_available_(
                best_cached_sum,
                L,
                R,
                (int8_t)(depth_budget - 1),
                (int8_t)std::max(0, (int)depth_budget - 2),
                *pkLp,
                *pkRp,
                left_best,
                right_best
            );
        }

        int ans = best_sum;

        if (best_cached_sum != std::numeric_limits<int>::max()) {
            ans = std::min(ans, best_cached_sum);
        }

        tighten_with_trie_min_if_available_(ans, kmask, depth_budget);

        if (proxy_caching_enabled) {
            cache_lickety_if_true_(
                kmask,
                depth_budget,
                KTAG,
                ans,
                /*allow_cache=*/true
            );
        }

        return ans;
    }

    // what greedy should choose for objective-minimizing split at depth 1, not based on information gain.
    // we also record what split this is in the cache. we don't utilize this in our experiments but support it.
    GreedyObjFirstSplit depth1_exact_solver_cached_continuous_with_first_split_(
        const Packed& mask,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        constexpr int8_t depth_budget = 1;

        K2 key{0, depth_budget};

        if (proxy_caching_enabled) {
            key.k = key_of_subproblem(mask, pk);

            if (auto it = greedy_first_split_cache.find(key);
                it != greedy_first_split_cache.end()) {
                return it->second;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos,
                bb_wrong
            );

            leaf_loss =
                leaf_objective_binary_from_counts(
                    n_sub,
                    pos,
                    bb_wrong
                );
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            GreedyObjFirstSplit out{leaf_loss, -1};

            if (proxy_caching_enabled) {
                greedy_first_split_cache.emplace(key, out);
            }

            return out;
        }

        int best_sum = leaf_loss;
        int best_feat = -1;

        Packed L(n_words), R(n_words);

        const int F = n_features;
        const int first_cont = first_continuous_feature_();
        ContinuousPath pi_cur = materialize_continuous_path_(cpath);

        // ordinary binary features
        for (int f = 0; f < first_cont; ++f) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[f], L, R, left_n);

                if (left_n == 0 || left_n == n_sub) {
                    continue;
                }
            } else {
                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);

                if (!L.any() || !R.any()) {
                    continue;
                }
            }

            const int sum = leaf_objective(L) + leaf_objective(R);

            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
            }
        }

        // continuous threshold groups
        for (int cont_pos = 0;
            cont_pos < (int)continuous_starts.size();
            ++cont_pos) {
            const int raw_start = continuous_starts[(size_t)cont_pos];
            const int raw_end = continuous_group_end_(cont_pos);

            if (raw_start >= F) {
                continue;
            }

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start,
                    std::min(raw_end, F),
                    pi_cur
                );

            if (start_idx >= end_idx) {
                continue;
            }

            ContinuousBestSplitResult cres =
                search_continuous_feature_for_best_split_continuous_(
                    mask,
                    /*depth_budget=*/1,
                    /*child_k=*/-1,
                    /*cache_lookup_k=*/(int8_t)std::max(0, (int)depth_budget - 2),
                    pk,
                    pi_cur,
                    start_idx,
                    end_idx,
                    best_sum,
                    ContinuousEvalMode::Exact
                );

            if (cres.best_feat >= 0 && cres.best_sum < best_sum) {
                best_sum = cres.best_sum;
                best_feat = cres.best_feat;
            }
        }

        GreedyObjFirstSplit out{best_sum, best_feat};

        if (proxy_caching_enabled) {
            greedy_first_split_cache.emplace(key, out);
        }

        return out;
    }

    struct GainSplitResult {
        double score = -std::numeric_limits<double>::infinity();
        int feat = -1;
    };

    inline double split_score_from_counts_binary_(
        int n,
        int pos,
        int nL,
        int posL
    ) const {
        const int nR = n - nL;
        if (n <= 0 || nL <= 0 || nR <= 0) {
            return -std::numeric_limits<double>::infinity();
        }

        const int posR = pos - posL;

        const double left_H  = entropy((double)posL / (double)nL);
        const double right_H = entropy((double)posR / (double)nR);

        // parent entropy is constant, so maximize negative weighted child entropy.
        return -((double)nL / (double)n) * left_H
            -((double)nR / (double)n) * right_H;
    }

    inline double split_score_from_counts_multiclass_(
        const std::vector<int>& parent_counts,
        const std::vector<int>& left_counts,
        int n,
        int nL
    ) const {
        const int nR = n - nL;
        if (n <= 0 || nL <= 0 || nR <= 0) {
            return -std::numeric_limits<double>::infinity();
        }

        std::vector<int> right_counts((size_t)num_classes, 0);
        for (int c = 0; c < num_classes; ++c) {
            right_counts[(size_t)c] =
                parent_counts[(size_t)c] - left_counts[(size_t)c];
        }

        const double left_H  = entropy_multiclass(left_counts, nL);
        const double right_H = entropy_multiclass(right_counts, nR);

        // parent entropy is constant, so maximize negative weighted child entropy.
        return -((double)nL / (double)n) * left_H
            -((double)nR / (double)n) * right_H;
    }

    GainSplitResult best_binary_score_split_(
        const Packed& mask,
        int first_feat,
        int end_feat,
        int n_total,
        int pos_total,
        int bb_wrong_total
    ) const {
        GainSplitResult best;

        if (first_feat >= end_feat) {
            return best;
        }

        const Packed& Ypos =
            Y_bits[(size_t)1];

        const bool objective_mode =
            greedy_split_mode == 2;

        const bool count_deferral_errors =
            objective_mode && use_deferral;

        for (int f = first_feat; f < end_feat; ++f) {
            const Packed& Xf =
                X_bits[(size_t)f];

            int nL = 0;
            int posL = 0;
            int left_bb_wrong = 0;

            for (int w = 0; w < n_words; ++w) {
                const uint64_t left_bits =
                    mask.w[(size_t)w] &
                    Xf.w[(size_t)w];

                nL += popcnt64(left_bits);

                posL += popcnt64(
                    left_bits &
                    Ypos.w[(size_t)w]
                );

                if (count_deferral_errors) {
                    left_bb_wrong += popcnt64(
                        left_bits &
                        BBwrong.w[(size_t)w]
                    );
                }
            }

            if (nL == 0 || nL == n_total) {
                continue;
            }

            double score = 0.0;

            if (objective_mode) {
                const int nR =
                    n_total - nL;

                const int posR =
                    pos_total - posL;

                const int right_bb_wrong =
                    bb_wrong_total -
                    left_bb_wrong;

                const int left_loss =
                    leaf_objective_binary_from_counts(
                        nL,
                        posL,
                        left_bb_wrong
                    );

                const int right_loss =
                    leaf_objective_binary_from_counts(
                        nR,
                        posR,
                        right_bb_wrong
                    );

                score =
                    -static_cast<double>(
                        left_loss + right_loss
                    );
            } else {
                score =
                    split_score_from_counts_binary_(
                        n_total,
                        pos_total,
                        nL,
                        posL
                    );
            }

            if (score > best.score) {
                best.score = score;
                best.feat = f;
            }
        }

        return best;
    }
    
    // general scoring method
    GainSplitResult best_continuous_score_split_(
        const Packed& mask,
        int start_idx,
        int end_idx,
        int n_total,
        int pos_total,
        int bb_wrong_total
    ) const {
        GainSplitResult best;

        if (start_idx >= end_idx) return best;

        const Packed& Ypos = Y_bits[(size_t)1];

        int prev_nL = -1;
        int prev_posL = -1;
        int prev_left_bb_wrong = -1;

        const bool objective_mode =
            greedy_split_mode == 2;

        const bool count_deferral_errors =
            objective_mode && use_deferral;

        for (int feat = start_idx; feat < end_idx; ++feat) {
            const Packed& Xf = X_bits[(size_t)feat];

            int nL = 0;
            int posL = 0;
            int left_bb_wrong = 0;

            for (int w = 0; w < n_words; ++w) {
                const uint64_t left_bits =
                    mask.w[(size_t)w] & Xf.w[(size_t)w];

                nL += popcnt64(left_bits);
                posL += popcnt64(left_bits & Ypos.w[(size_t)w]);
                if (count_deferral_errors) {
                    left_bb_wrong += popcnt64(
                        left_bits &
                        BBwrong.w[(size_t)w]
                    );
                }
            }

            // same active split as previous threshold in this subproblem.
            if (
                nL == prev_nL &&
                posL == prev_posL &&
                (
                    !count_deferral_errors ||
                    left_bb_wrong ==
                        prev_left_bb_wrong
                )
            ) {
                continue;
            }

            prev_nL = nL;
            prev_posL = posL;
            prev_left_bb_wrong =
                left_bb_wrong;

            if (nL == 0 || nL == n_total) continue;

            double score = 0.0;

            if (objective_mode) {
                const int nR =
                    n_total - nL;

                const int posR =
                    pos_total - posL;

                const int right_bb_wrong =
                    bb_wrong_total -
                    left_bb_wrong;

                const int left_loss =
                    leaf_objective_binary_from_counts(
                        nL,
                        posL,
                        left_bb_wrong
                    );

                const int right_loss =
                    leaf_objective_binary_from_counts(
                        nR,
                        posR,
                        right_bb_wrong
                    );

                score =
                    -static_cast<double>(
                        left_loss + right_loss
                    );
            } else {
                score =
                    split_score_from_counts_binary_(
                        n_total,
                        pos_total,
                        nL,
                        posL
                    );
            }

            if (score > best.score) {
                best.score = score;
                best.feat = feat;
            }
        }

        return best;
    }

    GainSplitResult best_binary_score_split_multiclass_(
        const Packed& mask,
        int first_feat,
        int end_feat,
        int n_total,
        int bb_wrong_total,
        const std::vector<int>& parent_counts
    ) const {
        GainSplitResult best;

        if (first_feat >= end_feat) {
            return best;
        }

        const bool objective_mode =
            greedy_split_mode == 2;

        const bool count_deferral_errors =
            objective_mode && use_deferral;

        std::vector<int> left_counts(
            (size_t)num_classes,
            0
        );

        std::vector<int> right_counts(
            (size_t)num_classes,
            0
        );

        for (int f = first_feat; f < end_feat; ++f) {
            std::fill(
                left_counts.begin(),
                left_counts.end(),
                0
            );

            const Packed& Xf =
                X_bits[(size_t)f];

            int nL = 0;
            int left_bb_wrong = 0;

            for (int w = 0; w < n_words; ++w) {
                const uint64_t left_bits =
                    mask.w[(size_t)w] &
                    Xf.w[(size_t)w];

                nL += popcnt64(left_bits);

                for (int c = 0; c < num_classes; ++c) {
                    left_counts[(size_t)c] +=
                        popcnt64(
                            left_bits &
                            Y_bits[(size_t)c]
                                .w[(size_t)w]
                        );
                }

                if (count_deferral_errors) {
                    left_bb_wrong += popcnt64(
                        left_bits &
                        BBwrong.w[(size_t)w]
                    );
                }
            }

            if (nL == 0 || nL == n_total) {
                continue;
            }

            double score = 0.0;

            if (objective_mode) {
                for (int c = 0; c < num_classes; ++c) {
                    right_counts[(size_t)c] =
                        parent_counts[(size_t)c] -
                        left_counts[(size_t)c];
                }

                const int right_bb_wrong =
                    bb_wrong_total -
                    left_bb_wrong;

                const int left_loss =
                    leaf_objective_multiclass_from_counts_(
                        left_counts,
                        left_bb_wrong
                    );

                const int right_loss =
                    leaf_objective_multiclass_from_counts_(
                        right_counts,
                        right_bb_wrong
                    );

                score =
                    -static_cast<double>(
                        left_loss + right_loss
                    );
            } else {
                score =
                    split_score_from_counts_multiclass_(
                        parent_counts,
                        left_counts,
                        n_total,
                        nL
                    );
            }

            if (score > best.score) {
                best.score = score;
                best.feat = f;
            }
        }

        return best;
    }

    GainSplitResult best_continuous_score_split_multiclass_(
        const Packed& mask,
        int start_idx,
        int end_idx,
        int n_total,
        int bb_wrong_total,
        const std::vector<int>& parent_counts
    ) const {
        GainSplitResult best;

        if (start_idx >= end_idx) {
            return best;
        }

        const bool objective_mode =
            greedy_split_mode == 2;

        const bool count_deferral_errors =
            objective_mode && use_deferral;

        std::vector<int> left_counts(
            (size_t)num_classes,
            0
        );

        std::vector<int> right_counts(
            (size_t)num_classes,
            0
        );

        std::vector<int> prev_left_counts(
            (size_t)num_classes,
            -1
        );

        int prev_nL = -1;
        int prev_left_bb_wrong = -1;

        for (
            int feat = start_idx;
            feat < end_idx;
            ++feat
        ) {
            std::fill(
                left_counts.begin(),
                left_counts.end(),
                0
            );

            const Packed& Xf =
                X_bits[(size_t)feat];

            int nL = 0;
            int left_bb_wrong = 0;

            for (int w = 0; w < n_words; ++w) {
                const uint64_t left_bits =
                    mask.w[(size_t)w] &
                    Xf.w[(size_t)w];

                nL += popcnt64(left_bits);

                for (
                    int c = 0;
                    c < num_classes;
                    ++c
                ) {
                    left_counts[(size_t)c] +=
                        popcnt64(
                            left_bits &
                            Y_bits[(size_t)c]
                                .w[(size_t)w]
                        );
                }

                if (count_deferral_errors) {
                    left_bb_wrong +=
                        popcnt64(
                            left_bits &
                            BBwrong.w[(size_t)w]
                        );
                }
            }

            if (
                nL == prev_nL &&
                left_counts == prev_left_counts &&
                (
                    !count_deferral_errors ||
                    left_bb_wrong ==
                        prev_left_bb_wrong
                )
            ) {
                continue;
            }

            prev_nL = nL;
            prev_left_counts = left_counts;
            prev_left_bb_wrong =
                left_bb_wrong;

            if (nL == 0 || nL == n_total) {
                continue;
            }

            double score = 0.0;

            if (objective_mode) {
                for (
                    int c = 0;
                    c < num_classes;
                    ++c
                ) {
                    right_counts[(size_t)c] =
                        parent_counts[(size_t)c] -
                        left_counts[(size_t)c];
                }

                const int right_bb_wrong =
                    bb_wrong_total -
                    left_bb_wrong;

                const int left_loss =
                    leaf_objective_multiclass_from_counts_(
                        left_counts,
                        left_bb_wrong
                    );

                const int right_loss =
                    leaf_objective_multiclass_from_counts_(
                        right_counts,
                        right_bb_wrong
                    );

                score =
                    -static_cast<double>(
                        left_loss + right_loss
                    );
            } else {
                score =
                    split_score_from_counts_multiclass_(
                        parent_counts,
                        left_counts,
                        n_total,
                        nL
                    );
            }

            if (score > best.score) {
                best.score = score;
                best.feat = feat;
            }
        }

        return best;
    }

    

    // get the sorted lists of indices to do greedy more efficiently for large-scale problems
    int greedy_numerical_entry_point(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk
    ) {
        if (numerical_X_cols_for_greedy.size() != continuous_starts.size() ||
            numerical_global_sorted_idx.size() != continuous_starts.size() ||
            numerical_unique_values_for_greedy.size() != continuous_starts.size()) {
            throw std::logic_error(
                "Numerical greedy representation is not aligned with continuous_starts."
            );
        }

        if (depth_budget <= 0) {
            return leaf_objective(mask);
        }

        // important: check the greedy cache before paying to build the
        // NumericalGreedyState.
        if (proxy_caching_enabled) {
            const uint64_t kmask = key_of_subproblem(mask, pk);
            const K2 key{kmask, depth_budget};

            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) {
                return it->second;
            }
        }

        NumericalGreedyState state;
        const std::size_t G = numerical_global_sorted_idx.size();
        state.sorted_idx_by_num.resize(G);

        // true:
        // collect active rows once, then sort those local rows separately for each numerical feature.
        //
        // false: scan each global pre-sorted row list and keep active rows.
        // O(N/64 + k n log n) vs O(k N)
        constexpr bool BUILD_STATE_BY_LOCAL_SORT = false;

        if constexpr (BUILD_STATE_BY_LOCAL_SORT) {
            std::vector<int> active_rows;
            active_rows.reserve((std::size_t)count_total(mask));

            for (int w = 0; w < n_words; ++w) {
                uint64_t bits = mask.w[(std::size_t)w];

                while (bits) {
#if defined(_MSC_VER)
                    unsigned long bidx;
                    _BitScanForward64(&bidx, bits);
                    const int b = static_cast<int>(bidx);
#else
                    const int b = __builtin_ctzll(bits);
#endif
                    const int row = (w << 6) + b;

                    if (row < n_samples) {
                        active_rows.push_back(row);
                    }

                    bits &= (bits - 1);
                }
            }

            for (std::size_t g = 0; g < G; ++g) {
                const std::vector<double>& x =
                    numerical_X_cols_for_greedy[g];

                auto& active_order = state.sorted_idx_by_num[g];
                active_order = active_rows;

                std::stable_sort(
                    active_order.begin(),
                    active_order.end(),
                    [&](int a, int b) {
                        const double xa = x[(std::size_t)a];
                        const double xb = x[(std::size_t)b];

                        if (xa < xb) return true;
                        if (xb < xa) return false;
                        return a < b;
                    }
                );
            }
        } else {
            for (std::size_t g = 0; g < G; ++g) {
                const auto& global_order = numerical_global_sorted_idx[g];
                auto& active_order = state.sorted_idx_by_num[g];

                active_order.clear();
                active_order.reserve(global_order.size());

                for (int row : global_order) {
                    if (mask_has_row_(mask, row)) {
                        active_order.push_back(row);
                    }
                }
            }
        }

        return train_greedy_continuous_numerical(
            mask,
            depth_budget,
            pk,
            state
        );
    }

    GainSplitResult best_numerical_score_split_(
        int num_group,
        const std::vector<int>& sorted_rows,
        int n_total,
        int pos_total,
        int bb_wrong_total,
        const std::vector<int>& parent_counts
    ) const {
        GainSplitResult best;
        best.score = -std::numeric_limits<double>::infinity();
        best.feat = -1;

        if (num_group < 0 || num_group >= (int)continuous_starts.size()) {
            return best;
        }

        if (sorted_rows.size() <= 1) {
            return best;
        }

        const int start_feat =
            continuous_starts[(std::size_t)num_group];

        const int end_feat =
            continuous_group_end_(num_group);

        if (start_feat >= end_feat) {
            return best;
        }

        const std::vector<double>& x =
            numerical_X_cols_for_greedy[(std::size_t)num_group];

        const std::vector<double>& vals =
            numerical_unique_values_for_greedy[(std::size_t)num_group];

        if (vals.size() <= 1) {
            return best;
        }

        const int expected_end_feat =
            start_feat + (int)vals.size() - 1;

        if (expected_end_feat != end_feat) {
            throw std::logic_error(
                "continuous_starts/end does not match "
                "numerical_unique_values_for_greedy."
            );
        }

        int nL = 0;
        int posL = 0;
        int left_bb_wrong = 0;

        std::vector<int> left_counts;

        if (num_classes != 2) {
            left_counts.assign(
                (std::size_t)num_classes,
                0
            );
        }

        std::size_t i = 0;

        while (i < sorted_rows.size()) {
            const double value =
                x[(std::size_t)sorted_rows[i]];

            // move the whole tie block with this value to the left.
            std::size_t j = i;

            while (
                j < sorted_rows.size() &&
                x[(std::size_t)sorted_rows[j]] == value
            ) {
                const int row = sorted_rows[j];

                ++nL;

                if (num_classes == 2) {
                    if (y_train[(std::size_t)row] == 1) {
                        ++posL;
                    }
                } else {
                    const int c =
                        y_train[(std::size_t)row];

                    if (c >= 0 && c < num_classes) {
                        ++left_counts[(std::size_t)c];
                    }
                }

                if (use_deferral && bb_wrong_at_(row)) {
                    ++left_bb_wrong;
                }

                ++j;
            }

            // cannot split after the maximum active value.
            if (j >= sorted_rows.size()) {
                break;
            }

            auto it =
                std::lower_bound(
                    vals.begin(),
                    vals.end(),
                    value
                );

            if (it == vals.end() || *it != value) {
                throw std::logic_error(
                    "Active numerical value not found in "
                    "global unique values."
                );
            }

            const int global_idx =
                (int)std::distance(
                    vals.begin(),
                    it
                );

            // last global value has no threshold column.
            if (global_idx + 1 >= (int)vals.size()) {
                break;
            }

            const int feat =
                start_feat + global_idx;

            if (feat < start_feat || feat >= end_feat) {
                throw std::logic_error(
                    "Mapped numerical threshold feature index "
                    "is out of group range."
                );
            }

            if (nL == 0 || nL == n_total) {
                i = j;
                continue;
            }

            double score = 0.0;

            if (num_classes == 2) {
                const int nR =
                    n_total - nL;

                const int posR =
                    pos_total - posL;

                if (greedy_split_mode == 2) {
                    const int right_bb_wrong =
                        bb_wrong_total - left_bb_wrong;

                    const int left_loss =
                        leaf_objective_binary_from_counts(
                            nL,
                            posL,
                            left_bb_wrong
                        );

                    const int right_loss =
                        leaf_objective_binary_from_counts(
                            nR,
                            posR,
                            right_bb_wrong
                        );

                    score =
                        -(double)(left_loss + right_loss);
                } else {
                    // information gain does not depend on black-box mistakes.
                    score =
                        split_score_from_counts_binary_(
                            n_total,
                            pos_total,
                            nL,
                            posL
                        );
                }
            } else {
                if (greedy_split_mode == 2) {
                    const int right_bb_wrong =
                        bb_wrong_total - left_bb_wrong;

                    const int left_loss =
                        leaf_objective_multiclass_from_counts_(
                            left_counts,
                            left_bb_wrong
                        );

                    std::vector<int> right_counts(
                        (std::size_t)num_classes,
                        0
                    );

                    for (int c = 0; c < num_classes; ++c) {
                        right_counts[(std::size_t)c] =
                            parent_counts[(std::size_t)c] -
                            left_counts[(std::size_t)c];
                    }

                    const int right_loss =
                        leaf_objective_multiclass_from_counts_(
                            right_counts,
                            right_bb_wrong
                        );

                    score =
                        -(double)(left_loss + right_loss);
                } else {
                    score =
                        split_score_from_counts_multiclass_(
                            parent_counts,
                            left_counts,
                            n_total,
                            nL
                        );
                }
            }

            if (score > best.score) {
                best.score = score;
                best.feat = feat;
            }

            i = j;
        }

        return best;
    }

    static inline double entropy_binary_count_(int n, int pos) {
        if (n <= 0) return 0.0;

        const double p = (double)pos / (double)n;

        if (p <= 0.0 || p >= 1.0) return 0.0;

        return -p * std::log(p) - (1.0 - p) * std::log(1.0 - p);
    }

    double binary_impurity_gain_score_(
        int n_total,
        int pos_total,
        int nL,
        int posL
    ) const {
        const int nR = n_total - nL;
        const int posR = pos_total - posL;

        if (nL <= 0 || nR <= 0) {
            return -std::numeric_limits<double>::infinity();
        }

        const double parent =
            entropy_binary_count_(n_total, pos_total);

        const double left =
            entropy_binary_count_(nL, posL);

        const double right =
            entropy_binary_count_(nR, posR);

        return parent
            - ((double)nL / (double)n_total) * left
            - ((double)nR / (double)n_total) * right;
    }

    int leaf_objective_multiclass_from_counts_(
        const std::vector<int>& counts,
        int bb_wrong
    ) const {
        int n = 0;
        int best = 0;

        for (int c : counts) {
            n += c;
            best = std::max(best, c);
        }

        const int predict_loss =
            gamma + (n - best);

        if (!use_deferral || n == 0) {
            return predict_loss;
        }

        const int defer_loss =
            gamma
            + defer_penalty_from_count_(n)
            + bb_wrong;

        return std::min(
            predict_loss,
            defer_loss
        );
    }

    void partition_numerical_state_(
        const NumericalGreedyState& parent_state,
        const Packed& L,
        const Packed& R,
        NumericalGreedyState& left_state,
        NumericalGreedyState& right_state
    ) const {
        const std::size_t G = parent_state.sorted_idx_by_num.size();

        const std::size_t nL = (std::size_t)L.count();
        const std::size_t nR = (std::size_t)R.count();

        left_state.sorted_idx_by_num.resize(G);
        right_state.sorted_idx_by_num.resize(G);

        for (std::size_t g = 0; g < G; ++g) {
            const auto& src = parent_state.sorted_idx_by_num[g];
            auto& dstL = left_state.sorted_idx_by_num[g];
            auto& dstR = right_state.sorted_idx_by_num[g];

            dstL.clear();
            dstR.clear();

            dstL.reserve(nL);
            dstR.reserve(nR);

            for (int row : src) {
                if (mask_has_row_(L, row)) {
                    dstL.push_back(row);
                } else {
                    dstR.push_back(row);
                }
            }
        }
    }

    // the actual training with the numerical representation, called state here, now that we have it
    // this does use the special depth 1 solver to get the optimal split at depth 1 instead of information-gain selection
    int train_greedy_continuous_numerical(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk,
        const NumericalGreedyState& state
    ) {
        if (depth_budget <= 0) {
            return leaf_objective(mask);
        }

        // at depth 1, modes 1/2 use exact stump optimization.
        if (
            depth_budget == 1 &&
            (
                greedy_split_mode == 1 ||
                greedy_split_mode == 2
            )
        ) {
            return depth1_numerical_solver_cached(
                mask,
                pk,
                state
            );
        }

        uint64_t kmask = 0;
        K2 key{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask =
                key_of_subproblem(mask, pk);

            key.k = kmask;

            if (
                auto it = greedy_cache.find(key);
                it != greedy_cache.end()
            ) {
                return it->second;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        std::vector<int> parent_counts;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos,
                bb_wrong
            );

            leaf_loss =
                leaf_objective_binary_from_counts(
                    n_sub,
                    pos,
                    bb_wrong
                );
        } else {
            n_sub = count_total(mask);
            count_per_class(mask, parent_counts);
            bb_wrong = count_bb_wrong(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (
                cache_cheap_subproblems &&
                proxy_caching_enabled
            ) {
                greedy_cache.emplace(
                    key,
                    leaf_loss
                );
            }

            return leaf_loss;
        }

        GainSplitResult best;
        best.score =
            -std::numeric_limits<double>::infinity();

        best.feat = -1;

        const int first_cont =
            first_continuous_feature_();

        // ordinary binary features still use the existing binary scan.
        // this helper must itself be deferral-aware when greedy_split_mode == 2.
        if (num_classes == 2) {
            GainSplitResult bres =
                best_binary_score_split_(
                    mask,
                    0,
                    first_cont,
                    n_sub,
                    pos,
                    bb_wrong
                );

            if (
                bres.feat >= 0 &&
                bres.score > best.score
            ) {
                best = bres;
            }
        } else {
            GainSplitResult bres =
                best_binary_score_split_multiclass_(
                    mask,
                    0,
                    first_cont,
                    n_sub,
                    bb_wrong,
                    parent_counts
                );

            if (
                bres.feat >= 0 &&
                bres.score > best.score
            ) {
                best = bres;
            }
        }

        // continuous features use numerical sorted lists.
        const int G =
            (int)continuous_starts.size();

        if (
            (int)state.sorted_idx_by_num.size() != G
        ) {
            throw std::logic_error(
                "Numerical greedy state is not aligned "
                "with continuous_starts."
            );
        }

        for (int g = 0; g < G; ++g) {
            const auto& sorted_rows =
                state.sorted_idx_by_num[
                    (std::size_t)g
                ];

            GainSplitResult cres =
                best_numerical_score_split_(
                    g,
                    sorted_rows,
                    n_sub,
                    pos,
                    bb_wrong,
                    parent_counts
                );

            if (
                cres.feat >= 0 &&
                cres.score > best.score
            ) {
                best = cres;
            }
        }

        if (best.feat < 0) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(
                    key,
                    leaf_loss
                );
            }

            return leaf_loss;
        }

        Packed L(n_words);
        Packed R(n_words);

        split_threshold_bits_(
            mask,
            best.feat,
            L,
            R
        );

        if (!L.any() || !R.any()) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(
                    key,
                    leaf_loss
                );
            }

            return leaf_loss;
        }

        const PathKey* pkLp =
            &empty_pk();

        const PathKey* pkRp =
            &empty_pk();

        PathKey pkL_local;
        PathKey pkR_local;

        make_child_pks_if_needed_(
            best.feat,
            pk,
            pkLp,
            pkRp,
            pkL_local,
            pkR_local
        );

        NumericalGreedyState left_state;
        NumericalGreedyState right_state;

        partition_numerical_state_(
            state,
            L,
            R,
            left_state,
            right_state
        );

        const int left_loss =
            train_greedy_continuous_numerical(
                L,
                depth_budget - 1,
                *pkLp,
                left_state
            );

        const int right_loss =
            train_greedy_continuous_numerical(
                R,
                depth_budget - 1,
                *pkRp,
                right_state
            );

        const int split_loss =
            left_loss + right_loss;

        const int ans =
            std::min(
                leaf_loss,
                split_loss
            );

        if (proxy_caching_enabled) {
            greedy_cache.emplace(
                key,
                ans
            );
        }

        return ans;
    }

    int depth1_numerical_solver_cached(
        const Packed& mask,
        const PathKey& pk,
        const NumericalGreedyState& state
    ) {
        constexpr int8_t DEPTH = 1;

        uint64_t kmask = 0;
        K2 key{0, DEPTH};

        if (proxy_caching_enabled) {
            kmask =
                key_of_subproblem(mask, pk);

            key.k = kmask;

            if (
                auto it = greedy_cache.find(key);
                it != greedy_cache.end()
            ) {
                return it->second;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        std::vector<int> parent_counts;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos,
                bb_wrong
            );

            leaf_loss =
                leaf_objective_binary_from_counts(
                    n_sub,
                    pos,
                    bb_wrong
                );
        } else {
            n_sub = count_total(mask);
            count_per_class(mask, parent_counts);
            bb_wrong = count_bb_wrong(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (n_sub <= 1) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(
                    key,
                    leaf_loss
                );
            }

            return leaf_loss;
        }

        // any split has at least two leaves.
        if (leaf_loss <= 2 * gamma) {
            if (
                cache_cheap_subproblems &&
                proxy_caching_enabled
            ) {
                greedy_cache.emplace(
                    key,
                    leaf_loss
                );
            }

            return leaf_loss;
        }

        int best_sum = leaf_loss;

        Packed L(n_words);
        Packed R(n_words);

        const int first_cont =
            first_continuous_feature_();

        for (int f = 0; f < first_cont; ++f) {
            if (num_classes == 2) {
                int left_n = 0;

                split_bits_count_left(
                    mask,
                    X_bits[f],
                    L,
                    R,
                    left_n
                );

                if (
                    left_n == 0 ||
                    left_n == n_sub
                ) {
                    continue;
                }
            } else {
                and_bits(
                    mask,
                    X_bits[f],
                    L
                );

                andnot_bits(
                    mask,
                    X_bits[f],
                    R
                );

                if (!L.any() || !R.any()) {
                    continue;
                }
            }

            const int sum =
                leaf_objective(L) +
                leaf_objective(R);

            if (sum < best_sum) {
                best_sum = sum;
            }
        }

        const int G =
            (int)continuous_starts.size();

        if (
            (int)state.sorted_idx_by_num.size() != G
        ) {
            throw std::logic_error(
                "Numerical greedy state is not aligned "
                "with continuous_starts."
            );
        }

        if (
            (int)numerical_X_cols_for_greedy.size() != G ||
            (int)numerical_unique_values_for_greedy.size() != G
        ) {
            throw std::logic_error(
                "Numerical greedy arrays are not aligned "
                "with continuous_starts."
            );
        }

        for (int g = 0; g < G; ++g) {
            const int candidate =
                depth1_numerical_feature_best_sum_(
                    g,
                    state.sorted_idx_by_num[
                        (std::size_t)g
                    ],
                    n_sub,
                    pos,
                    bb_wrong,
                    parent_counts,
                    best_sum
                );

            if (candidate < best_sum) {
                best_sum = candidate;
            }
        }

        if (proxy_caching_enabled) {
            greedy_cache.emplace(
                key,
                best_sum
            );
        }

        return best_sum;
    } 

    // implementation of the fast scanning described in the appendix
    int depth1_numerical_feature_best_sum_(
        int num_group,
        const std::vector<int>& sorted_rows,
        int n_total,
        int pos_total,
        int bb_wrong_total,
        const std::vector<int>& parent_counts,
        int incumbent
    ) const {
        if (
            num_group < 0 ||
            num_group >= (int)continuous_starts.size()
        ) {
            return incumbent;
        }

        if (sorted_rows.size() <= 1) {
            return incumbent;
        }

        const int start_feat =
            continuous_starts[(std::size_t)num_group];

        const int end_feat =
            continuous_group_end_(num_group);

        if (start_feat >= end_feat) {
            return incumbent;
        }

        const std::vector<double>& x =
            numerical_X_cols_for_greedy[
                (std::size_t)num_group
            ];

        const std::vector<double>& vals =
            numerical_unique_values_for_greedy[
                (std::size_t)num_group
            ];

        if (vals.size() <= 1) {
            return incumbent;
        }

        const int expected_end_feat =
            start_feat + (int)vals.size() - 1;

        if (expected_end_feat != end_feat) {
            throw std::logic_error(
                "continuous_starts/end does not match "
                "numerical_unique_values_for_greedy."
            );
        }

        int best_sum = incumbent;

        int nL = 0;
        int posL = 0;
        int left_bb_wrong = 0;

        std::vector<int> left_counts;

        if (num_classes != 2) {
            left_counts.assign(
                (std::size_t)num_classes,
                0
            );
        }

        std::size_t i = 0;

        while (i < sorted_rows.size()) {
            const double value =
                x[(std::size_t)sorted_rows[i]];

            // Move the whole tie block for this value left.
            std::size_t j = i;

            while (
                j < sorted_rows.size() &&
                x[(std::size_t)sorted_rows[j]] == value
            ) {
                const int row =
                    sorted_rows[j];

                ++nL;

                if (num_classes == 2) {
                    if (
                        y_train[(std::size_t)row] == 1
                    ) {
                        ++posL;
                    }
                } else {
                    const int c =
                        y_train[(std::size_t)row];

                    if (
                        c < 0 ||
                        c >= num_classes
                    ) {
                        throw std::logic_error(
                            "Class label out of range in "
                            "numerical depth-1 solver."
                        );
                    }

                    ++left_counts[
                        (std::size_t)c
                    ];
                }

                if (
                    use_deferral &&
                    bb_wrong_at_(row)
                ) {
                    ++left_bb_wrong;
                }

                ++j;
            }

            // If all active rows are left, right child is empty.
            if (j >= sorted_rows.size()) {
                break;
            }

            auto it =
                std::lower_bound(
                    vals.begin(),
                    vals.end(),
                    value
                );

            if (it == vals.end() || *it != value) {
                throw std::logic_error(
                    "Active numerical value not found in "
                    "global unique values."
                );
            }

            const int global_idx =
                (int)std::distance(
                    vals.begin(),
                    it
                );

            if (global_idx + 1 >= (int)vals.size()) {
                break;
            }

            const int feat =
                start_feat + global_idx;

            if (
                feat < start_feat ||
                feat >= end_feat
            ) {
                throw std::logic_error(
                    "Mapped numerical threshold feature index "
                    "is out of group range."
                );
            }

            if (nL == 0 || nL == n_total) {
                i = j;
                continue;
            }

            int sum = 0;

            const int right_bb_wrong =
                bb_wrong_total - left_bb_wrong;

            if (num_classes == 2) {
                const int nR =
                    n_total - nL;

                const int posR =
                    pos_total - posL;

                const int left_loss =
                    leaf_objective_binary_from_counts(
                        nL,
                        posL,
                        left_bb_wrong
                    );

                const int right_loss =
                    leaf_objective_binary_from_counts(
                        nR,
                        posR,
                        right_bb_wrong
                    );

                sum =
                    left_loss + right_loss;
            } else {
                std::vector<int> right_counts(
                    (std::size_t)num_classes,
                    0
                );

                for (int c = 0; c < num_classes; ++c) {
                    right_counts[(std::size_t)c] =
                        parent_counts[(std::size_t)c] -
                        left_counts[(std::size_t)c];
                }

                const int left_loss =
                    leaf_objective_multiclass_from_counts_(
                        left_counts,
                        left_bb_wrong
                    );

                const int right_loss =
                    leaf_objective_multiclass_from_counts_(
                        right_counts,
                        right_bb_wrong
                    );

                sum =
                    left_loss + right_loss;
            }

            if (sum < best_sum) {
                best_sum = sum;
            }

            i = j;
        }

        return best_sum;
    }

    // essentially the same greedy solver was in praxis; there is a subtle optimization we add (proxy section of the appendix)
    int train_greedy_continuous(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (depth_budget <= 0) {
            return leaf_objective(mask);
        }

        // last split level: use exact stump optimization, continuous version
        if (depth_budget == 1 && (greedy_split_mode == 1 || greedy_split_mode == 2)) {
            return depthd_exact_proxy_objective_(mask, depth_budget, pk, cpath);
        }

        uint64_t kmask = 0;
        K2 key{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key.k = kmask;

            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) {
                return it->second;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        std::vector<int> parent_counts;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos,
                bb_wrong
            );

            leaf_loss =
                leaf_objective_binary_from_counts(
                    n_sub,
                    pos,
                    bb_wrong
                );
        } else {
            n_sub = count_total(mask);
            count_per_class(mask, parent_counts);
            bb_wrong = count_bb_wrong(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }

            return leaf_loss;
        }

        GainSplitResult best;
        best.score = -std::numeric_limits<double>::infinity();
        best.feat = -1;

        const int F = n_features;
        const int first_cont = first_continuous_feature_();

        // prdinary binary features.
        if (num_classes == 2) {
            GainSplitResult bres =
                best_binary_score_split_(
                    mask,
                    0,
                    first_cont,
                    n_sub,
                    pos,
                    bb_wrong
                );

            if (bres.feat >= 0 && bres.score > best.score) {
                best = bres;
            }
        } else {
            GainSplitResult bres =
                best_binary_score_split_multiclass_(
                    mask,
                    0,
                    first_cont,
                    n_sub,
                    bb_wrong,
                    parent_counts
                );

            if (bres.feat >= 0 && bres.score > best.score) {
                best = bres;
            }
        }

        // continuous threshold groups
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start = continuous_starts[(size_t)cont_pos];
            const int raw_end = continuous_group_end_(cont_pos);

            if (raw_start >= F) continue;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start,
                    std::min(raw_end, F),
                    cpath
                );

            if (start_idx >= end_idx) continue;

            GainSplitResult cres;

            if (num_classes == 2) {
                cres = best_continuous_score_split_(
                    mask,
                    start_idx,
                    end_idx,
                    n_sub,
                    pos,
                    bb_wrong
                );
            } else {
                cres = best_continuous_score_split_multiclass_(
                    mask,
                    start_idx,
                    end_idx,
                    n_sub,
                    bb_wrong,
                    parent_counts
                );
            }

            if (cres.feat >= 0 && cres.score > best.score) {
                best = cres;
            }
        }

        if (best.feat < 0) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }

            return leaf_loss;
        }

        Packed L(n_words), R(n_words);
        split_threshold_bits_(mask, best.feat, L, R);

        if (!L.any() || !R.any()) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }

            return leaf_loss;
        }


        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();

        PathKey pkL_local;
        PathKey pkR_local;

        make_child_pks_if_needed_(
            best.feat,
            pk,
            pkLp,
            pkRp,
            pkL_local,
            pkR_local
        );

        const ContinuousPath* cpathLp = &cpath;
        const ContinuousPath* cpathRp = &cpath;

        ContinuousPath cpathL_local;
        ContinuousPath cpathR_local;

        make_child_continuous_paths_if_needed_(
            best.feat,
            cpath,
            cpathLp,
            cpathRp,
            cpathL_local,
            cpathR_local
        );

        const int left_loss = train_greedy_continuous(
            L,
            depth_budget - 1,
            *pkLp,
            *cpathLp
        );

        const int right_loss = train_greedy_continuous(
            R,
            depth_budget - 1,
            *pkRp,
            *cpathRp
        );

        const int split_loss = left_loss + right_loss;
        const int ans = std::min(leaf_loss, split_loss);

        if (proxy_caching_enabled) {
            greedy_cache.emplace(key, ans);
        }

        return ans;
    }

    // the only difference here is we log the first split in a cache
    GreedyObjFirstSplit train_greedy_continuous_with_first_split_(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (depth_budget <= 0) {
            return GreedyObjFirstSplit{leaf_objective(mask), -1};
        }

        // last split level: use exact stump optimization, continuous version
        if (depth_budget == 1 && (greedy_split_mode == 1 || greedy_split_mode == 2)) {
            return depth1_exact_solver_cached_continuous_with_first_split_(
                mask,
                pk,
                cpath
            );
        }

        uint64_t kmask = 0;
        K2 key{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key.k = kmask;

            if (auto it = greedy_first_split_cache.find(key);
                it != greedy_first_split_cache.end()) {
                return it->second;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        std::vector<int> parent_counts;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos,
                bb_wrong
            );

            leaf_loss =
                leaf_objective_binary_from_counts(
                    n_sub,
                    pos,
                    bb_wrong
                );
        } else {
            n_sub = count_total(mask);
            count_per_class(mask, parent_counts);
            bb_wrong = count_bb_wrong(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            GreedyObjFirstSplit out{leaf_loss, -1};
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                greedy_first_split_cache.emplace(key, out);
            }

            return out;
        }

        GainSplitResult best;
        best.score = -std::numeric_limits<double>::infinity();
        best.feat = -1;

        const int F = n_features;
        const int first_cont = first_continuous_feature_();

        // prdinary binary features.
        if (num_classes == 2) {
            GainSplitResult bres =
                best_binary_score_split_(
                    mask,
                    0,
                    first_cont,
                    n_sub,
                    pos,
                    bb_wrong
                );

            if (bres.feat >= 0 && bres.score > best.score) {
                best = bres;
            }
        } else {
            GainSplitResult bres =
                best_binary_score_split_multiclass_(
                    mask,
                    0,
                    first_cont,
                    n_sub,
                    bb_wrong,
                    parent_counts
                );

            if (bres.feat >= 0 && bres.score > best.score) {
                best = bres;
            }
        }

        // continuous threshold groups
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start = continuous_starts[(size_t)cont_pos];
            const int raw_end = continuous_group_end_(cont_pos);

            if (raw_start >= F) continue;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start,
                    std::min(raw_end, F),
                    cpath
                );

            if (start_idx >= end_idx) continue;

            GainSplitResult cres;

            if (num_classes == 2) {
                cres = best_continuous_score_split_(
                    mask,
                    start_idx,
                    end_idx,
                    n_sub,
                    pos,
                    bb_wrong
                );
            } else {
                cres = best_continuous_score_split_multiclass_(
                    mask,
                    start_idx,
                    end_idx,
                    n_sub,
                    bb_wrong,
                    parent_counts
                );
            }

            if (cres.feat >= 0 && cres.score > best.score) {
                best = cres;
            }
        }

        if (best.feat < 0) {
            GreedyObjFirstSplit out{leaf_loss, -1};
            if (proxy_caching_enabled) {
                greedy_first_split_cache.emplace(key, out);
            }

            return out;
        }

        Packed L(n_words), R(n_words);
        split_threshold_bits_(mask, best.feat, L, R);

        if (!L.any() || !R.any()) {
            GreedyObjFirstSplit out{leaf_loss, -1};
            if (proxy_caching_enabled) {
                greedy_first_split_cache.emplace(key, out);
            }

            return out;
        }

        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();

        PathKey pkL_local;
        PathKey pkR_local;

        make_child_pks_if_needed_(
            best.feat,
            pk,
            pkLp,
            pkRp,
            pkL_local,
            pkR_local
        );

        const ContinuousPath* cpathLp = &cpath;
        const ContinuousPath* cpathRp = &cpath;

        ContinuousPath cpathL_local;
        ContinuousPath cpathR_local;

        make_child_continuous_paths_if_needed_(
            best.feat,
            cpath,
            cpathLp,
            cpathRp,
            cpathL_local,
            cpathR_local
        );

        const int left_loss = train_greedy_continuous_with_first_split_(
            L,
            depth_budget - 1,
            *pkLp,
            *cpathLp
        ).obj;

        const int right_loss = train_greedy_continuous_with_first_split_(
            R,
            depth_budget - 1,
            *pkRp,
            *cpathRp
        ).obj;

        const int split_loss = left_loss + right_loss;
        const int ans = std::min(leaf_loss, split_loss);

        GreedyObjFirstSplit out{ans, best.feat};
        if (proxy_caching_enabled) {
            greedy_first_split_cache.emplace(key, out); 
        }

        return out;
    }

    // end continuous algorithms 

    // these are now existing binary algorithms - nothing new exists beyond this point. our contribution ends here.

    int train_greedy(const Packed& mask, int8_t depth_budget, const PathKey& pk) {
        if (depth_budget == 0) {
            return leaf_objective(mask);
        }
        if (depth_budget == 1 && (greedy_split_mode == 1 || greedy_split_mode == 2)) {
            return depthd_exact_proxy_objective_(mask, 1, pk); 
        }

        uint64_t kmask = 0;
        K2 key{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key.k = kmask;
            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) return it->second; // the objective of the tree returned by the proxy
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos,
                bb_wrong
            );

            leaf_loss =
                leaf_objective_binary_from_counts(
                    n_sub,
                    pos,
                    bb_wrong
                );
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }
            return leaf_loss;
        }

        // decide which split-selection heuristic to use
        bool use_entropy;
        if (greedy_split_mode == 0) {
            use_entropy = true;                  // always entropy-driven
        } else if (greedy_split_mode == 1) {
            use_entropy = (depth_budget != 1);   // special depth==1 solver
        } else { // greedy_split_mode == 2
            use_entropy = false;                 // always minimize child leaf objective
        }


        // choose split via entropy gain
        int best_feat;
        if (num_classes == 2) {
           best_feat = find_best_split_binary_known_counts(
                mask,
                n_sub,
                pos,
                bb_wrong,
                use_entropy
            );
        } else {
            best_feat = find_best_split(mask, use_entropy); // TODO
        }
        if (best_feat < 0) {
            if (proxy_caching_enabled) greedy_cache.emplace(key, leaf_loss);
            return leaf_loss;
        }
        
        Packed L(n_words), R(n_words);
        and_bits(mask, X_bits[best_feat], L);
        andnot_bits(mask, X_bits[best_feat], R);
       if (!L.any() || !R.any()) {
            throw std::logic_error(
                "Invalid split choice in greedy tree method"
            );
        }
        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();
        PathKey pkL_local, pkR_local;
        make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

        int left_obj  = train_greedy(L, depth_budget - 1, *pkLp);
        int right_obj = train_greedy(R, depth_budget - 1, *pkRp);
        int split_obj = left_obj + right_obj;

        int ans = min(leaf_loss, split_obj);
        if (proxy_caching_enabled) greedy_cache.emplace(key, ans);
        return ans;
    }

    int eval_with_lookahead(const Packed& m, int depth, int k, const PathKey& pk) {
        if (k <= 0) return greedy_proxy_objective_(m, depth, pk);
        return generalized_lickety_split(m, depth, k, pk);
    }


    inline int next_k_cycle(int8_t k) const {
            // cycle: ... 3->2->1->K->K-1->...
            return (k > 1) ? (k - 1) : lookahead_init;
        }

        inline void split_bits_count_left(
        const Packed& mask,
        const Packed& split,
        Packed& L,
        Packed& R,
        int& left_n
    ) const {
        left_n = popcount_and_make_split_words(
            mask.w.data(),
            split.w.data(),
            L.w.data(),
            R.w.data(),
            n_words,
            tail_mask
        );
    }

    int depth1_exact_solver_cached(const Packed& mask, const PathKey& pk) {
        constexpr int8_t DEPTH = 1;
        constexpr int8_t KTAG  = 0;

        const bool use_greedy_cache_for_depth1 =
            should_route_continuous_lickety_depth1_to_binary_greedy_();

        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);

            if (use_greedy_cache_for_depth1) {
                auto it = greedy_cache.find(K2{kmask, DEPTH});
                if (it != greedy_cache.end()) return it->second;
            } else {
                if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
            }
        }

        auto cache_depth1 = [&](int val, bool allow_cache) {
            if (!proxy_caching_enabled || !allow_cache) return;

            if (use_greedy_cache_for_depth1) {
                greedy_cache.emplace(K2{kmask, DEPTH}, val);
            } else {
                cache_lickety_if_true_(kmask, DEPTH, KTAG, val, /*allow_cache=*/true);
            }
        };

        if (num_classes == 2) {
            int n_sub = 0;
            int pos_total = 0;
            int bb_wrong_total = 0;

            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos_total,
                bb_wrong_total
            );

            const int leaf_loss =
                leaf_objective_binary_from_counts(
                    n_sub,
                    pos_total,
                    bb_wrong_total
                );

            // only cache cheap subproblems if flag enabled
            if (leaf_loss <= 2 * gamma) {
                cache_depth1(leaf_loss, cache_cheap_subproblems);
                return leaf_loss;
            }

            int best_sum = std::numeric_limits<int>::max();

            const Packed& Ypos = Y_bits[(size_t)1];
            const auto& feats = proxy_features_for_(ProxyLoopKind::DepthDExact);

            auto eval_feature = [&](int f) {
                const Packed& Xf = X_bits[(size_t)f];

                int left_n = 0;
                int left_pos = 0;
                int left_bb_wrong = 0;

                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw =
                        mask.w[(size_t)i] &
                        Xf.w[(size_t)i];

                    left_n += popcnt64(lw);

                    left_pos += popcnt64(
                        lw & Ypos.w[(size_t)i]
                    );

                    if (use_deferral) {
                        left_bb_wrong += popcnt64(
                            lw & BBwrong.w[(size_t)i]
                        );
                    }
                }

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                const int right_pos =
                    pos_total - left_pos;

                const int right_bb_wrong =
                    bb_wrong_total - left_bb_wrong;

                const int sum =
                    leaf_objective_binary_from_counts(
                        left_n,
                        left_pos,
                        left_bb_wrong
                    )
                    +
                    leaf_objective_binary_from_counts(
                        right_n,
                        right_pos,
                        right_bb_wrong
                    );

                if (sum < best_sum) {
                    best_sum = sum;
                }
            };

            
            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) eval_feature(f);
            } else {
                for (int f : feats) eval_feature(f);
            }

            int ans = leaf_loss;
            if (best_sum != std::numeric_limits<int>::max()) {
                ans = std::min(ans, best_sum);
            }

            cache_depth1(ans, /*allow_cache=*/true);
            return ans;
        }

        const int leaf_loss = leaf_objective(mask);

        // only cache cheap subproblems if flag enabled
        if (leaf_loss <= 2 * gamma) {
            cache_depth1(leaf_loss, cache_cheap_subproblems);
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const auto& feats = proxy_features_for_(ProxyLoopKind::DepthDExact);

        auto eval_feature = [&](int f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) return;

            const int sum = leaf_objective(L) + leaf_objective(R);
            if (sum < best_sum) best_sum = sum;
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) eval_feature(f);
        } else {
            for (int f : feats) eval_feature(f);
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) {
            ans = std::min(ans, best_sum);
        }

        cache_depth1(ans, /*allow_cache=*/true);
        return ans;
    }


    int depth2_special_solver_cached(const Packed& mask, const PathKey& pk){
        constexpr int8_t DEPTH = 2;
        constexpr int8_t KTAG  = 1;

        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos,
                bb_wrong
            );

            leaf_loss =
                leaf_objective_binary_from_counts(
                    n_sub,
                    pos,
                    bb_wrong
                );
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const auto& feats = proxy_features_for_(ProxyLoopKind::DepthDExact);

        auto eval_feature = [&](int f) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[f], L, R, left_n);
                if (left_n == 0 || left_n == n_sub) return;
            } else {
                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);
                if (!L.any() || !R.any()) return;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_best  = depth1_exact_solver_cached(L, *pkLp);
            const int right_best = depth1_exact_solver_cached(R, *pkRp);
            const int sum = left_best + right_best;

            if (sum < best_sum) best_sum = sum;
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) eval_feature(f);
        } else {
            for (int f : feats) eval_feature(f);
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (proxy_caching_enabled) cache_lickety_if_true_(kmask, DEPTH, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }

    int depth2_fixed_root_children_best_sum_bitvector_(
        const Packed& rootL,
        const Packed& rootR,
        const std::vector<int>& feats,
        int incumbent
    ) const {
        const int leafL = leaf_objective(rootL);
        const int leafR = leaf_objective(rootR);

        int bestL = leafL;
        int bestR = leafR;

        int nLroot = 0;
        int nRroot = 0;

        if (num_classes == 2) {
            int pos_tmp = 0;
            count_total_pos_binary(rootL, nLroot, pos_tmp);
            count_total_pos_binary(rootR, nRroot, pos_tmp);
        } else {
            nLroot = count_total(rootL);
            nRroot = count_total(rootR);
        }

        // if the one-leaf child is already <= 2 * gamma, no depth-1 split under that child can strictly improve it.
        const bool scanL = (nLroot > 1 && leafL > 2 * gamma);
        const bool scanR = (nRroot > 1 && leafR > 2 * gamma);

        if (!scanL && !scanR) {
            return bestL + bestR;
        }

        Packed LL(n_words), LR(n_words);
        Packed RL(n_words), RR(n_words);

        auto eval_second_feature = [&](int f2) {
            if (scanL) {
                if (num_classes == 2) {
                    int left_n = 0;
                    split_bits_count_left(rootL, X_bits[(std::size_t)f2], LL, LR, left_n);

                    if (left_n != 0 && left_n != nLroot) {
                        const int candL =
                            leaf_objective(LL) + leaf_objective(LR);

                        if (candL < bestL) {
                            bestL = candL;
                        }
                    }
                } else {
                    and_bits(rootL, X_bits[(std::size_t)f2], LL);
                    andnot_bits(rootL, X_bits[(std::size_t)f2], LR);

                    if (LL.any() && LR.any()) {
                        const int candL =
                            leaf_objective(LL) + leaf_objective(LR);

                        if (candL < bestL) {
                            bestL = candL;
                        }
                    }
                }
            }

            if (scanR) {
                if (num_classes == 2) {
                    int left_n = 0;
                    split_bits_count_left(rootR, X_bits[(std::size_t)f2], RL, RR, left_n);

                    if (left_n != 0 && left_n != nRroot) {
                        const int candR =
                            leaf_objective(RL) + leaf_objective(RR);

                        if (candR < bestR) {
                            bestR = candR;
                        }
                    }
                } else {
                    and_bits(rootR, X_bits[(std::size_t)f2], RL);
                    andnot_bits(rootR, X_bits[(std::size_t)f2], RR);

                    if (RL.any() && RR.any()) {
                        const int candR =
                            leaf_objective(RL) + leaf_objective(RR);

                        if (candR < bestR) {
                            bestR = candR;
                        }
                    }
                }
            }
        };

        if (feats.empty()) {
            for (int f2 = 0; f2 < n_features; ++f2) {
                eval_second_feature(f2);

                // for a fixed nonempty root split, the final subtree has at least one leaf on each side, so 2 * gamma is a lower bound.
                if (bestL + bestR <= 2 * gamma) {
                    break;
                }
            }
        } else {
            for (int f2 : feats) {
                eval_second_feature(f2);

                if (bestL + bestR <= 2 * gamma) {
                    break;
                }
            }
        }

        (void)incumbent;
        return bestL + bestR;
    }


    int depthd_exact_solver_cached(const Packed& mask, int8_t depth_budget, const PathKey& pk) {
        if (depth_budget <= 0) return leaf_objective(mask);
        if (depth_budget == 1) return depth1_exact_solver_cached(mask, pk);
        if (depth_budget == 2) return depth2_special_solver_cached(mask, pk);

        const int8_t DEPTH = depth_budget;
        const int8_t KTAG  = depth_budget - 1;


        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(
                mask,
                n_sub,
                pos,
                bb_wrong
            );

            leaf_loss =
                leaf_objective_binary_from_counts(
                    n_sub,
                    pos,
                    bb_wrong
                );
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();
        int best_cached_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const auto& feats = proxy_features_for_(ProxyLoopKind::DepthDExact);

        auto eval_feature = [&](int f) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[f], L, R, left_n);
                if (left_n == 0 || left_n == n_sub) return;
            } else {
                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);
                if (!L.any() || !R.any()) return;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_best  = depthd_exact_solver_cached(L, depth_budget - 1, *pkLp);
            const int right_best = depthd_exact_solver_cached(R, depth_budget - 1, *pkRp);

            const int sum = left_best + right_best;
            if (sum < best_sum) best_sum = sum;

            update_cached_rollout_sum_if_available_(
                best_cached_sum,
                L,
                R,
                (int8_t)(depth_budget - 1),
                (int8_t)std::max(0, (int)depth_budget - 2),
                *pkLp,
                *pkRp,
                left_best,
                right_best
            );
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) eval_feature(f);
        } else {
            for (int f : feats) eval_feature(f);
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (best_cached_sum != std::numeric_limits<int>::max()) {
            ans = std::min(ans, best_cached_sum);
        }

        tighten_with_trie_min_if_available_(ans, kmask, depth_budget);

        if (proxy_caching_enabled) cache_lickety_if_true_(kmask, depth_budget, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }


    inline void cache_lickety_if_true_(uint64_t kmask, int8_t depth_budget, int8_t k, int val, bool allow_cache) {
        if (!allow_cache) return;
        const bool use_kla = use_kla_cache();
        if (use_kla) lickety_cache_kla.emplace(KLA{kmask, depth_budget, k}, val);
        else         lickety_cache_k2.emplace(K2 {kmask, depth_budget},     val);
    }

    inline bool try_get_lickety_cached_(uint64_t kmask, int8_t depth_budget, int8_t k, int& out_val) const {
        const bool use_kla = use_kla_cache();
        if (use_kla) {
            auto it = lickety_cache_kla.find(KLA{kmask, depth_budget, k});
            if (it == lickety_cache_kla.end()) return false;
            out_val = it->second;
            return true;
        } else {
            auto it = lickety_cache_k2.find(K2{kmask, depth_budget});
            if (it == lickety_cache_k2.end()) return false;
            out_val = it->second;
            return true;
        }
    }

    inline bool try_get_trie_min_objective_(
        uint64_t kmask,
        int8_t depth_budget,
        int& out_val
    ) const {
        if (!trie_cache_enabled) return false;

        auto it = trie_cache.find(K2{kmask, depth_budget});
        if (it == trie_cache.end()) return false;
        if (!it->second) return false;

        const int v = it->second->min_objective;
        if (v == std::numeric_limits<int>::max()) return false;

        out_val = v;
        return true;
    }

    inline void tighten_with_trie_min_if_available_(
        int& ans,
        uint64_t kmask,
        int8_t depth_budget
    ) const {
        if (!stronger_rollout) return;

        int trie_min = 0;
        if (try_get_trie_min_objective_(kmask, depth_budget, trie_min)) {
            ans = std::min(ans, trie_min);
        }
    }

    inline void tighten_with_child_lickety_cache_sum_(
        int& best_sum,
        const Packed& L,
        const Packed& R,
        int8_t child_depth,
        int8_t lookup_k,
        const PathKey& pkL,
        const PathKey& pkR,
        int fallback_left,
        int fallback_right
    ) const {
        if (!stronger_rollout || !proxy_caching_enabled) return;

        int left_val = fallback_left;
        int right_val = fallback_right;

        bool have_left = false;
        bool have_right = false;

        const uint64_t kL = key_of_subproblem(L, pkL);
        const uint64_t kR = key_of_subproblem(R, pkR);

        have_left = try_get_lickety_cached_(kL, child_depth, lookup_k, left_val);
        have_right = try_get_lickety_cached_(kR, child_depth, lookup_k, right_val);

        if (have_left || have_right) {
            best_sum = std::min(best_sum, left_val + right_val);
        }
    }

    inline void update_cached_rollout_sum_if_available_(
        int& best_cached_sum,
        const Packed& L,
        const Packed& R,
        int8_t child_depth,
        int8_t lookup_k,
        const PathKey& pkL,
        const PathKey& pkR,
        int fallback_left,
        int fallback_right
    ) const {
        if (!stronger_rollout || !proxy_caching_enabled) return;

        int left_val = fallback_left;
        int right_val = fallback_right;

        bool have_left = false;
        bool have_right = false;

        const uint64_t kL = key_of_subproblem(L, pkL);
        const uint64_t kR = key_of_subproblem(R, pkR);

        have_left  = try_get_lickety_cached_(kL, child_depth, lookup_k, left_val);
        have_right = try_get_lickety_cached_(kR, child_depth, lookup_k, right_val);

        if (have_left || have_right) {
            best_cached_sum = std::min(best_cached_sum, left_val + right_val);
        }
    }

    // our modified lickety_split algorithm that is O(nk^2d^2). 
    int lickety_split_k1(const Packed& mask, int8_t depth_budget, const PathKey& pk)
    {
        if (depth_budget == 0) return leaf_objective(mask);
        const bool depthd_mode_matches_lickety = use_restricted_depthd_exact_proxy_();

        if (depth_budget == 1 && depthd_mode_matches_lickety) {
                return depthd_exact_proxy_objective_(mask, 1, pk);
        }

        if (depth_budget == 2 && depthd_mode_matches_lickety) {
                return depthd_exact_proxy_objective_(mask, 2, pk);
        }

        // (k=1 so use K2 cache)
        uint64_t kmask = 0;
        K2 key2{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key2.k = kmask;
            if (auto it = lickety_cache_k2.find(key2); it != lickety_cache_k2.end())
                return it->second;
        }

        const int leaf_loss = leaf_objective(mask);

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                lickety_cache_k2.emplace(key2, leaf_loss);
            }
            return leaf_loss;
        }

        int best_feat = -1;
        int best_sum  = std::numeric_limits<int>::max();
        int best_cached_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        Packed bestL(n_words), bestR(n_words);

        const auto& feats = proxy_features_for_(ProxyLoopKind::Lickety);
        const int8_t child_depth = (int8_t)(depth_budget - 1);

        auto eval_feature = [&](int f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) return;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_greedy =
                greedy_proxy_objective_(L, child_depth, *pkLp);

            const int right_greedy =
                greedy_proxy_objective_(R, child_depth, *pkRp);

            const int sum = left_greedy + right_greedy;

            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
                bestL.w = L.w;
                bestR.w = R.w;
            }

            update_cached_rollout_sum_if_available_(
                best_cached_sum,
                L,
                R,
                child_depth,
                /*lookup_k=*/1,
                *pkLp,
                *pkRp,
                left_greedy,
                right_greedy
            );
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) {
                eval_feature(f);
            }
        } else {
            for (int f : feats) {
                eval_feature(f);
            }

            maybe_eval_continuous_greedy_suggested_split_(
                mask,
                depth_budget,
                pk,
                feats,
                eval_feature
            );
        }

        int ans = leaf_loss;

        if (best_cached_sum != std::numeric_limits<int>::max()) {
            ans = std::min(ans, best_cached_sum);
        }

        // recurse with constant k=1 (proxy_style=0 behavior)
        if (best_feat >= 0) {
            const int8_t next_depth = (int8_t)(depth_budget - 1);

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_loss  = lickety_split_k1(bestL, next_depth, *pkLp);
            const int right_loss = lickety_split_k1(bestR, next_depth, *pkRp);

            ans = std::min(ans, left_loss + right_loss);
            ans = std::min(ans, best_sum);
        }

        tighten_with_trie_min_if_available_(ans, kmask, depth_budget);

        if (proxy_caching_enabled) {
            lickety_cache_k2.emplace(key2, ans);
        }
        return ans;
    }

    // a generalized lickety_split algorithm with a lookahead parameter k. we also support other proxy styles here (such as recursively applying split) that have the same flavor.
    int generalized_lickety_split(const Packed& mask, int8_t depth_budget, int8_t k, const PathKey& pk) {
        // only do this call if also anytime is false to allow for proxy strength refinement
        if (!anytime_mode_active_ && k == 1 && lookahead_init == 1 && proxy_style == 0) {
            return lickety_split_k1(mask, depth_budget, pk);
        }

        const bool depthd_mode_matches_lickety = use_restricted_depthd_exact_proxy_();

        if (depth_budget == 0) {
            return leaf_objective(mask);
        }

        if (depth_budget == 1 && depthd_mode_matches_lickety) {
            return depthd_exact_proxy_objective_(mask, 1, pk); // if this operates over continuous, at least as good as binarized. if it is binarized, same thing. either way is good.
        }
       

        if (k > depth_budget - 1) k = depth_budget - 1;

        if (depth_budget == 2 && k == 1 && depthd_mode_matches_lickety) return depthd_exact_proxy_objective_(mask, 2, pk);
        if (k == depth_budget - 1 && depthd_mode_matches_lickety) {
            return depthd_exact_proxy_objective_(mask, depth_budget, pk);
        }

        uint64_t kmask = 0;
        K2  key2{0, depth_budget};
        KLA keyla{0, depth_budget, k};

        const bool use_kla = use_kla_cache();

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key2.k = kmask;
            keyla.k = kmask;

            if (use_kla) {
                if (auto it = lickety_cache_kla.find(keyla); it != lickety_cache_kla.end())
                    return it->second;
            } else {
                if (auto it = lickety_cache_k2.find(key2); it != lickety_cache_k2.end())
                    return it->second;
            }
        }

        const int leaf_loss = leaf_objective(mask);
        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                if (use_kla) lickety_cache_kla.emplace(keyla, leaf_loss);
                else         lickety_cache_k2.emplace(key2,  leaf_loss);
            }
            return leaf_loss;
        }

        int best_feat = -1;
        int best_sum  = numeric_limits<int>::max();
        int best_cached_sum = numeric_limits<int>::max();

        Packed L(n_words), R(n_words), bestL(n_words), bestR(n_words);

        const int child_k = k - 1;
        const auto& feats = proxy_features_for_(ProxyLoopKind::Lickety);

        auto eval_feature = [&](int f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) return;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_loss =
                eval_with_lookahead(L, depth_budget - 1, child_k, *pkLp);

            const int right_loss =
                eval_with_lookahead(R, depth_budget - 1, child_k, *pkRp);

            const int sum = left_loss + right_loss;

            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
            }

            update_cached_rollout_sum_if_available_(
                best_cached_sum,
                L,
                R,
                (int8_t)(depth_budget - 1),
                /*lookup_k=*/k,
                *pkLp,
                *pkRp,
                left_loss,
                right_loss
            );
            
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) {
                eval_feature(f);
            }
        } else {
            for (int f : feats) {
                eval_feature(f);
            }

            maybe_eval_continuous_greedy_suggested_split_(
                mask,
                depth_budget,
                pk,
                feats,
                eval_feature
            );
        }

        if (best_feat >= 0) {
            and_bits(mask, X_bits[best_feat], bestL);
            andnot_bits(mask, X_bits[best_feat], bestR);
        }

        int ans = leaf_loss;

        if (best_cached_sum != std::numeric_limits<int>::max()) {
            ans = std::min(ans, best_cached_sum);
        }

        int8_t k_recurse;

        if (proxy_style == 0) {
            // style 0: constant k (recursively choosing based on lower tier LicketySPLIT)
            k_recurse = k;
        } else if (proxy_style == 3) {
            // style 3: if we're running SPLIT without postprocessing, we don't need to do further recursive calls (the tree is fully determined).
            tighten_with_trie_min_if_available_(ans, kmask, depth_budget);
            ans = std::min(ans, best_sum);
            if (proxy_caching_enabled) {
                if (use_kla) lickety_cache_kla.emplace(keyla, ans);
                else         lickety_cache_k2.emplace(key2,  ans);
            }
            return ans;
        } else {
            // styles 1/2: restart when it hits 0 (recursively applying SPLIT, so we're cycling k, k-1, k-2, ... 2 1 k)
            k_recurse = (child_k == 0) ? lookahead_init : child_k;
        }


        if (best_feat >= 0) {
            const int next_depth = depth_budget - 1;

            int left_loss, right_loss;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

            left_loss  = generalized_lickety_split(bestL, next_depth, k_recurse, *pkLp);
            right_loss = generalized_lickety_split(bestR, next_depth, k_recurse, *pkRp);
            ans = std::min(ans, left_loss + right_loss); // do licketysplit and take the minimum of it and leaf, even if greedy doesn't perform better than the leaf.
            ans = std::min(ans, best_sum); // if greedy is allowed more features, or heuristic pruning happens, we never want to be worse than greedy

        }

        tighten_with_trie_min_if_available_(ans, kmask, depth_budget);

        if (proxy_caching_enabled) {
            if (use_kla) lickety_cache_kla.emplace(keyla, ans);
            else lickety_cache_k2.emplace(key2,  ans);
        }
        
        return ans;
    }

    int split_algorithm(const Packed& mask, int8_t depth_budget, int8_t k, const PathKey& pk) {
        if (depth_budget <= 0) return leaf_objective(mask);
        if (k > depth_budget - 1) k = depth_budget - 1; // same amount of computation is optimal
        if (k == depth_budget - 1) {
            return depthd_exact_proxy_objective_(mask, depth_budget, pk);
        }

        // if lookahead is exhausted, switch to optimal at this remaining depth
        if (k <= 0) {
            return depthd_exact_proxy_objective_(mask, depth_budget, pk);
        }

        const int8_t child_d = (int8_t)(depth_budget - 1);
        const int8_t child_k = (int8_t)(k - 1);

        Packed L(n_words), R(n_words);

        int best_feat = -1;
        int best_score = std::numeric_limits<int>::max();

        Packed bestL(n_words), bestR(n_words);
        PathKey bestPkL, bestPkR;
        bool have_best_pks = false;

        const auto& feats = proxy_features_for_(ProxyLoopKind::Lickety);

        auto eval_feature = [&](int f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) return;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            // choose the best split based on generalized_lickety_split at (d-1, k-1). with the strategy 3/4 (SPLIT algorithm), this is tracing out the splits that the SPLIT algorithm without postprocessing chose.
            int left_score  = lickety_proxy_objective_(L, child_d, child_k, *pkLp);
            int right_score = lickety_proxy_objective_(R, child_d, child_k, *pkRp);
            int score = left_score + right_score;

            if (score < best_score) {
                best_score = score;
                best_feat = f;

                bestL.w = L.w;
                bestR.w = R.w;

                if (key_mode == KeyMode::LITS_EXACT) {
                    bestPkL = *pkLp;
                    bestPkR = *pkRp;
                    have_best_pks = true;
                } else {
                    have_best_pks = false;
                }
            }
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) {
                eval_feature(f);
            }
        } else {
            for (int f : feats) {
                eval_feature(f);
            }
        }

        if (best_feat < 0) {
            return leaf_objective(mask);
        }

        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();
        PathKey pkL_local, pkR_local;

        if (key_mode == KeyMode::LITS_EXACT) {
            if (have_best_pks) {
                pkLp = &bestPkL;
                pkRp = &bestPkR;
            } else {
                make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);
            }
        }

        // instead of recursing with a greedy tree, we apply postprocessing here.
        // we do optimal on the children because we chose a split that was already best for greedy completion (aka we are tracing out the top k splits of the SPLIT tree without postprocessing to then apply it)
        if (child_k <= 0) {
            int left_cost  = depthd_exact_proxy_objective_(bestL, child_d, *pkLp);
            int right_cost = depthd_exact_proxy_objective_(bestR, child_d, *pkRp);
            return left_cost + right_cost;
        }

        // otherwise recurse using split() itself to find the next split it made
        int left_cost  = lickety_proxy_objective_(bestL, child_d, child_k, *pkLp);
        int right_cost = lickety_proxy_objective_(bestR, child_d, child_k, *pkRp);
        return left_cost + right_cost;
    }

    int find_best_split_binary_known_counts(
        const Packed& mask,
        int n_sub,
        int pos_total,
        int bb_wrong_total,
        bool use_entropy
    ) const {
        if (n_sub <= 1) return -1;

        const Packed& Ypos = Y_bits[(size_t)1];

        int best_f = -1;
        const auto& feats = proxy_features_for_(ProxyLoopKind::Greedy);

        if (use_entropy) {
            double best_score = 1e300;

            auto eval_feature = [&](int f) {
                int left_n = 0;
                int left_pos = 0;

                const Packed& Xf = X_bits[(size_t)f];

                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw =
                        mask.w[(size_t)i] & Xf.w[(size_t)i];

                    left_n += popcnt64(lw);
                    left_pos += popcnt64(
                        lw & Ypos.w[(size_t)i]
                    );
                }

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                const int right_pos = pos_total - left_pos;

                const double wl =
                    (double)left_n / (double)n_sub;
                const double wr =
                    (double)right_n / (double)n_sub;

                const double pl =
                    (double)left_pos / (double)left_n;
                const double pr =
                    (double)right_pos / (double)right_n;

                const double score =
                    wl * entropy(pl) + wr * entropy(pr);

                if (score < best_score) {
                    best_score = score;
                    best_f = f;
                }
            };

            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) {
                    eval_feature(f);
                }
            } else {
                for (int f : feats) {
                    eval_feature(f);
                }
            }

            return best_f;
        }

        int best_sum = std::numeric_limits<int>::max();

        auto eval_feature = [&](int f) {
            int left_n = 0;
            int left_pos = 0;
            int left_bb_wrong = 0;

            const Packed& Xf = X_bits[(size_t)f];

            if (use_deferral) {
                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw =
                        mask.w[(size_t)i] & Xf.w[(size_t)i];

                    left_n += popcnt64(lw);
                    left_pos += popcnt64(
                        lw & Ypos.w[(size_t)i]
                    );
                    left_bb_wrong += popcnt64(
                        lw & BBwrong.w[(size_t)i]
                    );
                }
            } else {
                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw =
                        mask.w[(size_t)i] & Xf.w[(size_t)i];

                    left_n += popcnt64(lw);
                    left_pos += popcnt64(
                        lw & Ypos.w[(size_t)i]
                    );
                }
            }

            const int right_n = n_sub - left_n;
            if (left_n == 0 || right_n == 0) return;

            const int right_pos = pos_total - left_pos;
            const int right_bb_wrong =
                bb_wrong_total - left_bb_wrong;

            const int left_loss =
                leaf_objective_binary_from_counts(
                    left_n,
                    left_pos,
                    left_bb_wrong
                );

            const int right_loss =
                leaf_objective_binary_from_counts(
                    right_n,
                    right_pos,
                    right_bb_wrong
                );

            const int sum = left_loss + right_loss;

            if (sum < best_sum) {
                best_sum = sum;
                best_f = f;
            }
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) {
                eval_feature(f);
            }
        } else {
            for (int f : feats) {
                eval_feature(f);
            }
        }

        return best_f;
    }

    int find_best_split_binary(
        const Packed& mask,
        bool use_entropy
    ) const {
        int n_sub = 0;
        int pos_total = 0;
        int bb_wrong_total = 0;

        count_total_pos_bbwrong_binary(
            mask,
            n_sub,
            pos_total,
            bb_wrong_total
        );

        return find_best_split_binary_known_counts(
            mask,
            n_sub,
            pos_total,
            bb_wrong_total,
            use_entropy
        );
    }


    int find_best_split(const Packed& mask, bool use_entropy) const {
        if (num_classes == 2) {
            return find_best_split_binary(mask, use_entropy);
        }
        const int n_sub = count_total(mask);
        if (n_sub <= 1) return -1;

        Packed L(n_words);

        if (use_entropy) {
            // total class counts under mask
            std::vector<int> total_cnt((size_t)num_classes, 0);
            for (int c = 0; c < num_classes; ++c) {
                total_cnt[(size_t)c] = popcount_and(mask, Y_bits[(size_t)c]);
            }

            int best_f = -1;
            double best_score = 1e300;

            std::vector<int> left_cnt((size_t)num_classes, 0);
            std::vector<int> right_cnt((size_t)num_classes, 0);

            const auto& feats = proxy_features_for_(ProxyLoopKind::Greedy);

            auto eval_feature = [&](int f) {
                // L = mask & X_bits[f]
                for (int i = 0; i < n_words; ++i) L.w[i] = mask.w[i] & X_bits[(size_t)f].w[i];
                L.w[n_words - 1] &= tail_mask;

                const int left_n  = L.count();
                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                // left class counts: popcount_and(L, Y_bits[c])
                for (int c = 0; c < num_classes; ++c) {
                    left_cnt[(size_t)c] = popcount_and(L, Y_bits[(size_t)c]);
                }

                // right class counts = total - left (no need to build)
                for (int c = 0; c < num_classes; ++c) {
                    right_cnt[(size_t)c] = total_cnt[(size_t)c] - left_cnt[(size_t)c];
                }

                const double wl = (double)left_n  / (double)n_sub;
                const double wr = (double)right_n / (double)n_sub;

                const double Hl = entropy_multiclass(left_cnt,  left_n);
                const double Hr = entropy_multiclass(right_cnt, right_n);

                const double score = (wl * Hl + wr * Hr);
                if (score < best_score) { 
                    best_score = score; 
                    best_f = f; 
                }
            };

            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) {
                    eval_feature(f);
                }
            } else {
                for (int f : feats) {
                    eval_feature(f);
                }
            }

            return best_f;

        } else {
            // minimize child leaf objectives: leaf_objective(L)+leaf_objective(R)
            int best_f = -1;
            int best_sum = std::numeric_limits<int>::max();

            Packed R(n_words);
            const auto& feats = proxy_features_for_(ProxyLoopKind::Greedy);

            auto eval_feature = [&](int f) {
                // L = mask & X_bits[f]
                for (int i = 0; i < n_words; ++i) L.w[i] = mask.w[i] & X_bits[f].w[i]; 
                L.w[n_words-1] &= tail_mask;

                const int left_n = L.count();
                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                // R = mask & ~X_bits[f]
                for (int i = 0; i < n_words; ++i) R.w[i] = mask.w[i] & ~X_bits[f].w[i];
                R.w[n_words-1] &= tail_mask;

                const int sum = leaf_objective(L) + leaf_objective(R);
                if (sum < best_sum) { best_sum = sum; best_f = f; }
            };

            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) {
                    eval_feature(f);
                }
            } else {
                for (int f : feats) {
                    eval_feature(f);
                }
            }

            return best_f;
        }
    }

    

    struct TaoNodeWork_ {
        PredNode* node = nullptr;
        Packed mask;
        int depth = 0;
    };

    struct TaoStumpSolution_ {
        int feature = -1;
        bool flipped = false;
        int errors = std::numeric_limits<int>::max();
    };

    int predict_training_sample_from_prednode_(const PredNode* node, int row) const {
        const PredNode* cur = node;
        while (cur && cur->feature >= 0) {
            cur = training_value_(row, cur->feature)
                ? cur->left.get()
                : cur->right.get();
        }
        if (!cur) {
            throw std::runtime_error("Malformed PredNode tree during TAO refinement.");
        }
        if (cur->prediction == DEFER_PREDICTION) {
            // alternating_optimization currently rejects deferral mode, so this is
            // only a defensive guard.
            if (prepared_bb_pred.empty() || row >= (int)prepared_bb_pred.size()) {
                throw std::runtime_error(
                    "TAO encountered a defer leaf without training black-box predictions."
                );
            }
            return prepared_bb_pred[(size_t)row];
        }
        return cur->prediction;
    }

    int count_prednode_leaves_(const PredNode* node) const {
        if (!node) return 0;
        if (node->feature < 0) return 1;
        return count_prednode_leaves_(node->left.get())
             + count_prednode_leaves_(node->right.get());
    }

    int tree_training_mistakes_(const PredNode* tree) const {
        int mistakes = 0;
        for (int row = 0; row < n_samples; ++row) {
            mistakes += (
                predict_training_sample_from_prednode_(tree, row)
                != y_train[(size_t)row]
            );
        }
        return mistakes;
    }

    int tree_training_objective_raw_(const PredNode* tree) const {
        return tree_training_mistakes_(tree)
             + gamma * count_prednode_leaves_(tree);
    }

    void collect_tao_nodes_with_masks_(
        PredNode* node,
        const Packed& mask,
        int depth,
        std::vector<TaoNodeWork_>& out
    ) const {
        if (!node || node->feature < 0) return;

        out.push_back(TaoNodeWork_{node, mask, depth});

        Packed left_mask((size_t)n_words);
        Packed right_mask((size_t)n_words);
        and_bits(mask, X_bits[(size_t)node->feature], left_mask);
        andnot_bits(mask, X_bits[(size_t)node->feature], right_mask);

        if (left_mask.any()) {
            collect_tao_nodes_with_masks_(
                node->left.get(), left_mask, depth + 1, out
            );
        }
        if (right_mask.any()) {
            collect_tao_nodes_with_masks_(
                node->right.get(), right_mask, depth + 1, out
            );
        }
    }

    // exact depth-1 classifier for the temporary tao labels. `included` is the
    // subset participating in this node's classification problem and `want_left`
    // is the subset whose temporary label is 1. Every X_bits column is scanned,
    // so all ordinary binary features and every retained continuous threshold are
    // considered. There is intentionally no cache lookup or insertion here.
    TaoStumpSolution_ solve_tao_stump_exact_(
        const Packed& included,
        const Packed& want_left
    ) const {
        TaoStumpSolution_ best;

        const int total = included.count();
        if (total <= 0) return best;

        const int total_ones = want_left.count();
        const int total_zeros = total - total_ones;

        for (int f = 0; f < n_features; ++f) {
            const Packed& split = X_bits[(size_t)f];

            const int true_total = popcount_and(included, split);
            const int true_ones = popcount_and(want_left, split);
            const int true_zeros = true_total - true_ones;

            const int false_ones = total_ones - true_ones;
            const int false_zeros = total_zeros - true_zeros;

            // normal orientation: X_bits[f]==1 -> current left subtree.
            const int normal_errors = true_zeros + false_ones;

            // reversed orientation: X_bits[f]==1 -> current right subtree.
            const int flipped_errors = true_ones + false_zeros;

            auto consider = [&](int errors, bool flipped) {
                if (
                    errors < best.errors ||
                    (errors == best.errors &&
                     (best.feature < 0 || f < best.feature)) ||
                    (errors == best.errors && f == best.feature &&
                     best.flipped && !flipped)
                ) {
                    best.errors = errors;
                    best.feature = f;
                    best.flipped = flipped;
                }
            };

            consider(normal_errors, false);
            consider(flipped_errors, true);

            if (best.errors == 0) {
                // zero is globally optimal
                break;
            }
        }

        return best;
    }

    bool optimize_tao_node_(PredNode* node, const Packed& node_mask) const {
        if (!node || node->feature < 0 || !node->left || !node->right) return false;

        Packed included((size_t)n_words);
        Packed want_left((size_t)n_words);

        int current_errors = 0;

        // descendants are fixed while optimizing this node. build the temporary
        // classification problem directly from the two child-subtree predictions.
        for (int wi = 0; wi < n_words; ++wi) {
            uint64_t bits = node_mask.w[(size_t)wi];
            while (bits) {
#if defined(_MSC_VER)
                unsigned long bit_index = 0;
                _BitScanForward64(&bit_index, bits);
                const int bit = (int)bit_index;
#else
                const int bit = __builtin_ctzll(bits);
#endif
                const int row = (wi << 6) + bit;
                bits &= (bits - 1);
                if (row >= n_samples) continue;

                const int y = y_train[(size_t)row];
                const bool left_correct =
                    predict_training_sample_from_prednode_(node->left.get(), row) == y;
                const bool right_correct =
                    predict_training_sample_from_prednode_(node->right.get(), row) == y;

                // include iff exactly one child is correct.
                if (left_correct == right_correct) continue;

                included.w[(size_t)(row >> 6)] |= 1ULL << (row & 63);
                if (left_correct) {
                    want_left.w[(size_t)(row >> 6)] |= 1ULL << (row & 63);
                }

                const bool currently_goes_left = training_value_(row, node->feature);
                const bool wants_left_now = left_correct;
                current_errors += (currently_goes_left != wants_left_now);
            }
        }

        if (!included.any()) return false;

        const TaoStumpSolution_ best =
            solve_tao_stump_exact_(included, want_left);

        // strict improvement only
        if (best.feature < 0 || best.errors >= current_errors) return false;

        node->feature = best.feature;
        if (best.flipped) {
            std::swap(node->left, node->right);
        }

        return true;
    }

    // not used by ArborEnum Rashomon mode, to support giving single decision tree algorithm results in package.
    shared_ptr<PredNode> build_best_tree_from_caches(const Packed& mask, int8_t depth_budget, const PathKey& pk) const {
        const int INF = std::numeric_limits<int>::max();

        const int n_sub = count_total(mask);
        if (n_sub == 0) {
            auto t = make_shared<PredNode>();
            t->feature = -1;
            t->prediction = 0;
            return t;
        }

        const BestLeafAction leaf =
        best_leaf_action(mask);

        const int leaf_pred = leaf.prediction;
        const int leaf_loss = leaf.loss;

        if (depth_budget <= 0) {
            auto t = make_shared<PredNode>();
            t->feature = -1;
            t->prediction = leaf_pred;
            return t;
        }

        // helper: lookup min cached objective for (mask, depth, pk) across greedy + lickety caches
        auto best_cached_obj = [&](const Packed& m, int8_t d, const PathKey& pk_child) -> int {
            if (d < 0) return 0;
            if (d==0) return leaf_objective(m);
            if (!proxy_caching_enabled) return INF;

            const uint64_t km = key_of_subproblem(m, pk_child);

            int best = INF;

            // greedy cache: (subproblem, depth)
            {
                auto itg = greedy_cache.find(K2{km, d});
                if (itg != greedy_cache.end()) best = std::min(best, itg->second);
            }

            // lickety cache:
            if (use_kla_cache()) {
                // try all k = 0..d-1
                for (int kk = 0; kk <= (int)(d-1); ++kk) {
                    auto it = lickety_cache_kla.find(KLA{km, d, kk});
                    if (it != lickety_cache_kla.end()) best = std::min(best, it->second);
                }
            } else {
                // K2-form: no k needed
                auto it = lickety_cache_k2.find(K2{km, d});
                if (it != lickety_cache_k2.end()) best = std::min(best, it->second);
            }

            return best;
        };

        // choose best split using cached objectives
        const int8_t child_d = (int8_t)(depth_budget - 1);

        int best_feat = -1;
        int best_sum  = INF;

        Packed L(n_words), R(n_words), bestL(n_words), bestR(n_words);
        PathKey bestPkL, bestPkR;
        bool have_best_pks = false;

        const int F = n_features;
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_obj  = best_cached_obj(L, child_d, *pkLp);
            const int right_obj = best_cached_obj(R, child_d, *pkRp);
            if (left_obj == INF || right_obj == INF) continue;

            const int sum = left_obj + right_obj;
            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
                bestL.w = L.w;
                bestR.w = R.w;

                if (key_mode == KeyMode::LITS_EXACT) {
                    bestPkL = *pkLp;
                    bestPkR = *pkRp;
                    have_best_pks = true;
                } else {
                    have_best_pks = false;
                }
            }
        }

        // If no split is yielded or it doesn't beat leaf, return leaf.
        if (best_feat < 0 || best_sum >= leaf_loss) {
            auto t = make_shared<PredNode>();
            t->feature = -1;
            t->prediction = leaf_pred;
            return t;
        }

        // --- recurse on chosen split ---
        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();
        PathKey pkL_local, pkR_local;

        if (key_mode == KeyMode::LITS_EXACT) {
            if (have_best_pks) {
                pkLp = &bestPkL;
                pkRp = &bestPkR;
            } else {
                make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);
            }
        } else {
            // non-literal mode: pk is ignored by key_of_subproblem anyway, so keep empty_pk()
            pkLp = &empty_pk();
            pkRp = &empty_pk();
        }

        auto left_tree  = build_best_tree_from_caches(bestL, child_d, *pkLp);
        auto right_tree = build_best_tree_from_caches(bestR, child_d, *pkRp);

        auto t = make_shared<PredNode>();
        t->feature = best_feat;
        t->prediction = -1;
        t->left = left_tree;
        t->right = right_tree;
        return t;
    }

    shared_ptr<PredNode> get_ith_tree(uint64_t i) const {
        if (!result) {
            throw runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }
        
        // count_trees will ensure that the histograms are built at the root and every child node (by building them if they are not yet built)
        uint64_t total = result->count_trees();
        if (i >= total) {
            throw out_of_range("Tree index out of range in get_ith_tree");
        }

        uint64_t cum = 0;
        int target_obj = -1;
        uint64_t k_within = 0;

        // hist is sorted by objective ascending
        for (const auto& e : result->hist) {
            if (i < cum + e.cnt) {
                target_obj = e.obj;
                k_within = i - cum;
                break;
            }
            cum += e.cnt;
        }
        if (target_obj < 0) {
            throw runtime_error("Failed to locate objective bucket in get_ith_tree");
        }

        return get_kth_tree_with_objective(result.get(), target_obj, k_within);
    }

    shared_ptr<PredNode> get_kth_tree_with_objective(const TreeTrieNode* node, int target_obj, uint64_t k) const {
        if (!node) {
            throw runtime_error("Null node in get_kth_tree_with_objective");
        }

        // handle leaf-only trees at this node
        for (const auto& leaf : node->leaves) {
            if (leaf.loss == target_obj) {
                if (k == 0) {
                    auto t = make_shared<PredNode>();
                    t->feature = -1;
                    t->prediction = leaf.prediction;
                    return t;
                }
                --k;
            }
        }

        // handle splits
        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();

            // total_here = #trees under this split with exactly target_obj
            uint64_t total_here = 0;

            // for each L histogram entry, we go over it. R is sorted by obj, so we can binary search to find each r_obj that pairs to sum to exactly this target_obj.
            for (const auto& le : L->hist) {
                int l_obj = le.obj;
                uint64_t lc = le.cnt;
                int r_obj = target_obj - l_obj;
                // binary search r_obj in R->hist
                auto it = lower_bound(
                    R->hist.begin(), R->hist.end(),
                    HistEntry{r_obj, 0},
                    hist_less
                );
                if (it != R->hist.end() && it->obj == r_obj) {
                    uint64_t rc = it->cnt;
                    uint64_t pairs = lc * rc;

                    if (total_here + pairs > k) { // k is smaller than the culm amount in this split we've seen so far (for the first time), so we know that we want to recurse on this split (which was already known), with this particular l_obj and r_obj, but we also need what index within each objective to recurse 
                        uint64_t rel = k - total_here; // what index inside this block the tree lives (again, 0 indexed)
                        uint64_t left_idx  = rel / rc; // left contributes lc possibilities, right contributes rc, a cross product without filtering, this indexing scheme works to break ties
                        uint64_t right_idx = rel % rc;

                        auto left_tree  = get_kth_tree_with_objective(L, l_obj,  left_idx); // now we have all the information we need, recurse
                        auto right_tree = get_kth_tree_with_objective(R, r_obj, right_idx);

                        auto t = make_shared<PredNode>();
                        t->feature = split.feature;
                        t->prediction = -1;
                        t->left = left_tree;
                        t->right = right_tree;
                        return t;
                    }

                    total_here += pairs;
                }

            }
            // skip all trees from this split that achieve target_obj
            k -= total_here;
            
        }

        throw out_of_range("Index out of range for given objective in get_kth_tree_with_objective");
    }

    void predict_tree_recursive(
        const PredNode* node,
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& bb_pred_row,
        std::vector<uint8_t>& out,
        const std::vector<int>& idx,
        bool use_placeholder,
        int defer_placeholder
    ) const {
        if (!node) return;

        if (node->feature < 0) {
            if (
                node->prediction ==
                DEFER_PREDICTION
            ) {
                for (int row : idx) {
                    const int p =
                        use_placeholder
                            ? defer_placeholder
                            : bb_pred_row[(size_t)row];

                    out[(size_t)row] =
                        static_cast<uint8_t>(p);
                }
            } else {
                const uint8_t pred =
                    static_cast<uint8_t>(
                        node->prediction
                    );

                for (int row : idx) {
                    out[(size_t)row] = pred;
                }
            }

            return;
        }

        const int f = node->feature;

        std::vector<int> left_idx;
        std::vector<int> right_idx;

        left_idx.reserve(idx.size());
        right_idx.reserve(idx.size());

        for (int row : idx) {
            if (X_row_major[(size_t)row][(size_t)f]) {
                left_idx.push_back(row);
            } else {
                right_idx.push_back(row);
            }
        }

        if (!left_idx.empty()) {
            predict_tree_recursive(
                node->left.get(),
                X_row_major,
                bb_pred_row,
                out,
                left_idx,
                use_placeholder,
                defer_placeholder
            );
        }

        if (!right_idx.empty()) {
            predict_tree_recursive(
                node->right.get(),
                X_row_major,
                bb_pred_row,
                out,
                right_idx,
                use_placeholder,
                defer_placeholder
            );
        }
    }

    void collect_paths(const PredNode* node, std::vector<int>& current, std::vector<std::vector<int>>& paths, std::vector<int>& preds) const {
        if (!node) {
            throw std::logic_error("collect_paths: encountered null node");
        }

        // leaf: record this path and prediction
        if (node->feature < 0) {
            paths.push_back(current); // current starts empty and is appended to along the dfs
            preds.push_back(node->prediction);
            return;
        }

        int f = node->feature;
        // IMPORTANT: we have to switch to 1-indexing here so that +- for the 0th (1st) feature means something

        // go left (true) -> +f or rather f+1
        current.push_back(f+1);
        collect_paths(node->left.get(), current, paths, preds);
        current.pop_back(); // backtrack after we complete a path so we have 1 vector that is updated in a nice way throughout this

        // go right (false) -> -f (technically -(f+1))
        current.push_back(-(f+1));
        collect_paths(node->right.get(), current, paths, preds);
        current.pop_back();
    }

// whole trie prediction for RID

private:
    static inline void and_bits_eval(
        const Packed& a,
        const Packed& b,
        Packed& out,
        int n_words,
        uint64_t tail_mask
    ) {
        if (n_words <= 0) return;

    #if ArborEnum_USE_AVX512
        int i = 0;
        for (; i + 8 <= n_words; i += 8) {
            __m512i va = _mm512_loadu_si512((const void*)(a.w.data() + i));
            __m512i vb = _mm512_loadu_si512((const void*)(b.w.data() + i));
            __m512i vc = _mm512_and_si512(va, vb);
            _mm512_storeu_si512((void*)(out.w.data() + i), vc);
        }
        for (; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] & b.w[(size_t)i];
        }
    #else
        for (int i = 0; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] & b.w[(size_t)i];
        }
    #endif

        out.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    static inline void andnot_bits_eval(
        const Packed& a,
        const Packed& b,
        Packed& out,
        int n_words,
        uint64_t tail_mask
    ) {
        if (n_words <= 0) return;

    #if ArborEnum_USE_AVX512
        int i = 0;
        for (; i + 8 <= n_words; i += 8) {
            __m512i va = _mm512_loadu_si512((const void*)(a.w.data() + i));
            __m512i vb = _mm512_loadu_si512((const void*)(b.w.data() + i));

            // _mm512_andnot_si512(x, y) computes ~x & y.
            // So this is ~b & a = a & ~b.
            __m512i vc = _mm512_andnot_si512(vb, va);

            _mm512_storeu_si512((void*)(out.w.data() + i), vc);
        }
        for (; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] & ~b.w[(size_t)i];
        }
    #else
        for (int i = 0; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] & ~b.w[(size_t)i];
        }
    #endif

        out.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    static inline void or_bits_eval(
        const Packed& a,
        const Packed& b,
        Packed& out,
        int n_words,
        uint64_t tail_mask
    ) {
        if (n_words <= 0) return;

    #if ArborEnum_USE_AVX512
        int i = 0;
        for (; i + 8 <= n_words; i += 8) {
            __m512i va = _mm512_loadu_si512((const void*)(a.w.data() + i));
            __m512i vb = _mm512_loadu_si512((const void*)(b.w.data() + i));
            __m512i vc = _mm512_or_si512(va, vb);
            _mm512_storeu_si512((void*)(out.w.data() + i), vc);
        }
        for (; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] | b.w[(size_t)i];
        }
    #else
        for (int i = 0; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] | b.w[(size_t)i];
        }
    #endif

        out.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    static inline bool any_eval(const Packed& a) {
        const int n_words = (int)a.w.size();
        if (n_words <= 0) return false;

    #if ArborEnum_USE_AVX512
        int i = 0;
        __m512i accum = _mm512_setzero_si512();

        for (; i + 8 <= n_words; i += 8) {
            __m512i v = _mm512_loadu_si512((const void*)(a.w.data() + i));
            accum = _mm512_or_si512(accum, v);
        }

        // if any 64-bit lane is nonzero, this mask is nonzero.
        if (_mm512_test_epi64_mask(accum, accum) != 0) {
            return true;
        }

        for (; i < n_words; ++i) {
            if (a.w[(size_t)i]) return true;
        }

        return false;
    #else
        for (uint64_t t : a.w) {
            if (t) return true;
        }
        return false;
    #endif
    }

    static inline void clear_eval(Packed& a) {
        if (!a.w.empty()) {
            std::memset(a.w.data(), 0, a.w.size() * sizeof(uint64_t));
        }
    }

    static inline Packed zeros_eval(int n_words) {
        return Packed((size_t)n_words);
    }

    static inline Packed copy_eval_mask(
        const Packed& m,
        int n_words,
        uint64_t tail_mask
    ) {
        Packed out((size_t)n_words);

        if (n_words > 0) {
            std::memcpy(
                out.w.data(),
                m.w.data(),
                (size_t)n_words * sizeof(uint64_t)
            );

            out.w[(size_t)(n_words - 1)] &= tail_mask;
        }

        return out;
    }

    static inline PackedPredMulti zeros_predmulti(
        int n_words,
        int num_classes
    ) {
        PackedPredMulti pm;

        pm.by_class.assign(
            (size_t)num_classes,
            Packed((size_t)n_words)
        );

        pm.deferred =
            Packed((size_t)n_words);

        return pm;
    }

    static inline void clear_predmulti(
        PackedPredMulti& pm
    ) {
        for (auto& p : pm.by_class) {
            if (!p.w.empty()) {
                std::memset(
                    p.w.data(),
                    0,
                    p.w.size() *
                        sizeof(uint64_t)
                );
            }
        }

        if (!pm.deferred.w.empty()) {
            std::memset(
                pm.deferred.w.data(),
                0,
                pm.deferred.w.size() *
                    sizeof(uint64_t)
            );
        }
    }

    // OR-combine two multiclass prediction packs into out
    static inline void or_predmulti(
        const PackedPredMulti& a,
        const PackedPredMulti& b,
        PackedPredMulti& out,
        int n_words,
        uint64_t tail_mask
    ) {
        const int C = (int)a.by_class.size();

        for (int c = 0; c < C; ++c) {
            const Packed& ac =
                a.by_class[(size_t)c];

            const Packed& bc =
                b.by_class[(size_t)c];

            Packed& oc =
                out.by_class[(size_t)c];

            or_bits_eval(
                ac,
                bc,
                oc,
                n_words,
                tail_mask
            );
        }

        or_bits_eval(
            a.deferred,
            b.deferred,
            out.deferred,
            n_words,
            tail_mask
        );
    }

    static inline Packed
    build_eval_bb_wrong_bits_(
        const std::vector<int>& y_eval,
        const std::vector<int>& bb_pred_eval,
        int num_classes,
        int n_words,
        uint64_t tail_mask
    ) {
        if (
            bb_pred_eval.size() !=
            y_eval.size()
        ) {
            throw std::runtime_error(
                "Eval bb_pred has a different "
                "number of rows than Eval y."
            );
        }

        Packed wrong((size_t)n_words);

        for (
            int i = 0;
            i < static_cast<int>(y_eval.size());
            ++i
        ) {
            const int yi =
                y_eval[(size_t)i];

            const int bi =
                bb_pred_eval[(size_t)i];

            if (
                yi < 0 ||
                yi >= num_classes
            ) {
                throw std::runtime_error(
                    "Eval y contains a class "
                    "not seen during training."
                );
            }

            if (
                bi < 0 ||
                bi >= num_classes
            ) {
                throw std::runtime_error(
                    "Eval bb_pred contains a class "
                    "not seen during training."
                );
            }

            if (yi != bi) {
                wrong.w[(size_t)(i >> 6)] |=
                    (1ULL << (i & 63));
            }
        }

        if (n_words > 0) {
            wrong.w[
                (size_t)(n_words - 1)
            ] &= tail_mask;
        }

        return wrong;
    }


    // build packed feature columns for EVAL X, turning into column major
    static inline EvalCtx build_eval_ctx_(const std::vector<std::vector<uint8_t>>& X_row_major, int n_features_expected) {
        EvalCtx ctx;

        ctx.n_eval = (int)X_row_major.size();
        if (ctx.n_eval == 0) {
            ctx.n_words = 0;
            ctx.tail_mask = ~0ULL;
            return ctx;
        }

        const int d = (int)X_row_major[0].size();
        if (d != n_features_expected) {
            throw std::runtime_error("Eval X has different number of features than training.");
        }

        ctx.n_words = (ctx.n_eval + 63) / 64;
        ctx.tail_mask = (ctx.n_eval % 64) ? ((1ULL << (ctx.n_eval % 64)) - 1ULL) : ~0ULL;

        ctx.X_bits_eval.assign((size_t)d, Packed((size_t)ctx.n_words));

        for (int f = 0; f < d; ++f) {
            Packed &col = ctx.X_bits_eval[f];
            clear_eval(col);
            for (int i = 0; i < ctx.n_eval; ++i) {
                if (X_row_major[(size_t)i][(size_t)f]) {
                    col.w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
                }
            }
            col.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
        }

        return ctx;
    }

    // all 1s bitvector (for the evaluation passed in dataset not train)
    static inline Packed eval_root_mask_(int n_words, uint64_t tail_mask) {
        Packed m((size_t)n_words);
        if (n_words == 0) return m;
        for (int i = 0; i < n_words - 1; ++i) m.w[(size_t)i] = ~0ULL;
        m.w[(size_t)(n_words - 1)] = tail_mask;
        return m;
    }

    static inline std::vector<ObjBucketMulti> to_sorted_buckets_multi_(
        std::unordered_map<int, std::vector<PackedPredMulti>>& acc
    ) {
        std::vector<ObjBucketMulti> out;
        out.reserve(acc.size());
        for (auto &kv : acc) {
            ObjBucketMulti b;
            b.obj = kv.first;
            b.preds = std::move(kv.second);
            out.push_back(std::move(b));
        }
        std::sort(out.begin(), out.end(),
                [](const ObjBucketMulti& a, const ObjBucketMulti& b){ return a.obj < b.obj; });
        return out;
    }

    // core recursion: returns buckets of predictions grouped by objective for ALL trees rooted at node with obj <= budget.
    std::vector<ObjBucketMulti> collect_preds_by_obj_(
        const TreeTrieNode* node,
        int budget,
        const Packed& eval_mask, // does not decrease size, just gets sparser
        const EvalCtx& ctx
    ) const {
        if (!node) return {};
        if (budget < 0) return {};

        if (node->min_objective == std::numeric_limits<int>::max()) return {};
        if (node->min_objective > budget) return {};

        // accumulate as obj (training) -> list of preds on evaluation (Packed)
        //std::unordered_map<int, std::vector<Packed>> acc;
        std::unordered_map<int, std::vector<PackedPredMulti>> acc;
        // heuristic reserve
        const int max_objs = budget - node->min_objective + 1;
        acc.reserve((size_t)std::max(1, max_objs));

        // leaves at this node
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            PackedPredMulti pm = zeros_predmulti(ctx.n_words, num_classes);

            if (ctx.n_words > 0) {
                if (leaf.prediction == DEFER_PREDICTION) {
                    pm.deferred.w = eval_mask.w;
                    pm.deferred.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
                } else {
                    const int pc = leaf.prediction;
                    if (pc < 0 || pc >= num_classes) {
                        throw std::runtime_error("Leaf prediction is outside valid class range.");
                    }

                    pm.by_class[(size_t)pc].w = eval_mask.w;
                    pm.by_class[(size_t)pc].w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
                }
            }

            acc[leaf.loss].push_back(std::move(pm));
            
            // storing the predictions in the map with that objective.
        }

        // splits
        const int INF = std::numeric_limits<int>::max();

        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;
            if (minL == INF || minR == INF) continue;

            // cap child budgets using the other side's min objective so everything found will pair with exactly one subtree on the other side
            int bL = budget - minR;
            int bR = budget - minL;
            if (bL < 0 || bR < 0) continue;

            // also cap by the budgets actually used to build those trie nodes. should never change anything (assuming we do iterative budget refinement, otherwise this is needed for tightening).
            bL = std::min(bL, L->budget);
            bR = std::min(bR, R->budget);

            // evaluation dataset routing masks
            Packed Lmask((size_t)ctx.n_words), Rmask((size_t)ctx.n_words);
            if (ctx.n_words > 0) {
                and_bits_eval(eval_mask, ctx.X_bits_eval[(size_t)split.feature], Lmask, ctx.n_words, ctx.tail_mask);
                andnot_bits_eval(eval_mask, ctx.X_bits_eval[(size_t)split.feature], Rmask, ctx.n_words, ctx.tail_mask);
            }

            // recurse
            auto Lb = collect_preds_by_obj_(L, bL, Lmask, ctx); // these return sorted lists of objective bucket objects
            auto Rb = collect_preds_by_obj_(R, bR, Rmask, ctx);
            if (Lb.empty() || Rb.empty()) continue;

            // for filtering by <= budget, both Lb and Rb are sorted by obj.
            // we'll two-pointer for each left obj to find all right objs <= (budget - l_obj).
            // size_t r_hi = 0; // exclusive upper bound index in Rb
            // for (size_t li = 0; li < Lb.size(); ++li) {
            //     const int lo = Lb[li].obj; // smallest objective initially
            //     if (lo > budget) break;
            //     const int rem = budget - lo; // how far do we have to look

            //     //while (r_hi < Rb.size() && Rb[r_hi].obj <= rem) ++r_hi; // getting the first invalid index. never look past the remainder because RHS is also sorted
            //     // we could either go over Lb in reverse order or just do this 
            //     auto it_end = std::upper_bound(R_objs.begin(), R_objs.end(), rem);
            //     size_t r_hi = static_cast<size_t>(std::distance(R_objs.begin(), it_end));

            std::vector<int> R_objs;
            R_objs.reserve(Rb.size());
            for (const auto& rb : Rb) {
                R_objs.push_back(rb.obj); // don't need predictions here
            }

            for (size_t li = 0; li < Lb.size(); ++li) {
                const int lo = Lb[li].obj;
                if (lo > budget) break;

                const int rem = budget - lo;

                auto it_end = std::upper_bound(R_objs.begin(), R_objs.end(), rem);
                const size_t r_hi = static_cast<size_t>(
                    std::distance(R_objs.begin(), it_end)
                );
                            
                if (r_hi == 0) continue; // no right objs fit

                // cross product (filtered)
                for (size_t ri = 0; ri < r_hi; ++ri) { // r_hi is one past the last valid
                    const int ro = Rb[ri].obj; // objective value for that bucket
                    const int tot = lo + ro; // additivity of objectives

                    // combine each left pred with each right pred (disjoint masks so OR is correct)
                    const auto& Lpreds = Lb[li].preds;
                    const auto& Rpreds = Rb[ri].preds;

                    // reserve some space in this objective bucket to reduce reallocs
                    auto &dest = acc[tot]; // alias for simplicity
                    // rough reserve: only if currently empty
                    if (dest.empty()) {
                            dest.reserve(std::max(Lpreds.size(), Rpreds.size()));
                        }

                    for (const auto& lp : Lpreds) {
                        for (const auto& rp : Rpreds) {
                            PackedPredMulti comb = zeros_predmulti(ctx.n_words, num_classes);
                            if (ctx.n_words > 0) {
                                or_predmulti(lp, rp, comb, ctx.n_words, ctx.tail_mask);
                            }
                            dest.push_back(std::move(comb));
                        }
                    }

                }
            }
        }

        return to_sorted_buckets_multi_(acc);
    }

    struct ObjMistakeBucket {
        int obj;
        std::vector<int> mistakes; // one entry per tree at this objective
    };


    struct ObjDeferralBucket {
        int obj;
        std::vector<int> deferrals;
    };

 
    static inline std::vector<ObjDeferralBucket> to_sorted_deferral_buckets_(
        std::unordered_map<int, std::vector<int>>& acc
    ) {
        std::vector<ObjDeferralBucket> out;
        out.reserve(acc.size());

        for (auto& kv : acc) {
            ObjDeferralBucket b;
            b.obj = kv.first;
            b.deferrals = std::move(kv.second);
            out.push_back(std::move(b));
        }

        std::sort(out.begin(), out.end(),
            [](const ObjDeferralBucket& a, const ObjDeferralBucket& b) {
                return a.obj < b.obj;
            });

        return out;
    }

    std::vector<ObjDeferralBucket> collect_deferrals_by_obj_(
        const TreeTrieNode* node,
        int budget,
        const Packed& eval_mask,
        const EvalCtx& ctx
    ) const {
        if (!node) return {};
        if (budget < 0) return {};

        constexpr int INF = std::numeric_limits<int>::max();

        if (node->min_objective == INF) return {};
        if (node->min_objective > budget) return {};

        std::unordered_map<int, std::vector<int>> acc;

        const int max_objs = budget - node->min_objective + 1;
        acc.reserve((size_t)std::max(1, max_objs));

        // leaf alternatives
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            int n_def = 0;

            if (ctx.n_words > 0 && leaf.prediction == DEFER_PREDICTION) {
                n_def = count_eval_mask_(eval_mask, ctx.n_words);
            }

            acc[leaf.loss].push_back(n_def);
        }

        // split alternatives
        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;

            if (minL == INF || minR == INF) continue;

            int bL = budget - minR;
            int bR = budget - minL;

            if (bL < 0 || bR < 0) continue;

            bL = std::min(bL, L->budget);
            bR = std::min(bR, R->budget);

            if (bL < minL || bR < minR) continue;

            Packed Lmask((size_t)ctx.n_words);
            Packed Rmask((size_t)ctx.n_words);

            if (ctx.n_words > 0) {
                const Packed& Xf = ctx.X_bits_eval[(size_t)split.feature];

                for (int w = 0; w < ctx.n_words; ++w) {
                    const uint64_t mw = eval_mask.w[(size_t)w];
                    const uint64_t xw = Xf.w[(size_t)w];

                    Lmask.w[(size_t)w] = mw & xw;
                    Rmask.w[(size_t)w] = mw & ~xw;
                }

                Lmask.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
                Rmask.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
            }

            auto Lb = collect_deferrals_by_obj_(L, bL, Lmask, ctx);
            auto Rb = collect_deferrals_by_obj_(R, bR, Rmask, ctx);

            if (Lb.empty() || Rb.empty()) continue;

            std::vector<int> R_objs;
            R_objs.reserve(Rb.size());
            for (const auto& rb : Rb) {
                R_objs.push_back(rb.obj);
            }

            for (const auto& lb : Lb) {
                if (lb.obj > bL) continue;

                const int rem = budget - lb.obj;

                auto it_end = std::upper_bound(R_objs.begin(), R_objs.end(), rem);
                const int j_end = (int)std::distance(R_objs.begin(), it_end);

                for (int j = 0; j < j_end; ++j) {
                    const auto& rb = Rb[(size_t)j];

                    const int total_obj = lb.obj + rb.obj;
                    if (total_obj > budget) continue;

                    auto& vec = acc[total_obj];

                    for (int ld : lb.deferrals) {
                        for (int rd : rb.deferrals) {
                            vec.push_back(ld + rd);
                        }
                    }
                }
            }
        }

        return to_sorted_deferral_buckets_(acc);
    }



    static inline std::vector<ObjMistakeBucket> to_sorted_mistake_buckets_(
        std::unordered_map<int, std::vector<int>>& acc
    ) {
        std::vector<ObjMistakeBucket> out;
        out.reserve(acc.size());

        for (auto& kv : acc) {
            ObjMistakeBucket b;
            b.obj = kv.first;
            b.mistakes = std::move(kv.second);
            out.push_back(std::move(b));
        }

        std::sort(out.begin(), out.end(),
            [](const ObjMistakeBucket& a, const ObjMistakeBucket& b) {
                return a.obj < b.obj;
            });

        return out;
    }

    static inline std::vector<Packed> build_eval_y_bits_(
        const std::vector<int>& y_eval,
        int num_classes,
        int n_words,
        uint64_t tail_mask
    ) {
        std::vector<Packed> Y_eval((size_t)num_classes, Packed((size_t)n_words));

        for (int c = 0; c < num_classes; ++c) {
            clear_eval(Y_eval[(size_t)c]);
        }

        for (int i = 0; i < (int)y_eval.size(); ++i) {
            const int yi = y_eval[(size_t)i];
            if (yi < 0 || yi >= num_classes) {
                throw std::runtime_error("Eval y contains a class not seen during training.");
            }
            Y_eval[(size_t)yi].w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
        }

        if (n_words > 0) {
            for (int c = 0; c < num_classes; ++c) {
                Y_eval[(size_t)c].w[(size_t)(n_words - 1)] &= tail_mask;
            }
        }

        return Y_eval;
    }

    static inline int count_eval_mask_(const Packed& mask, int n_words) {
        int s = 0;
        for (int i = 0; i < n_words; ++i) {
            s += popcnt64(mask.w[(size_t)i]);
        }
        return s;
    }

    static inline int popcount_and_eval_(
        const Packed& a,
        const Packed& b,
        int n_words
    ) {
        int s = 0;
        for (int i = 0; i < n_words; ++i) {
            s += popcnt64(a.w[(size_t)i] & b.w[(size_t)i]);
        }
        return s;
    }

    std::vector<ObjMistakeBucket> collect_mistakes_by_obj_(
        const TreeTrieNode* node,
        int budget,
        const Packed& eval_mask,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval
    ) const {
        if (!node) return {};
        if (budget < 0) return {};

        constexpr int INF = std::numeric_limits<int>::max();

        if (node->min_objective == INF) return {};
        if (node->min_objective > budget) return {};

        // training objective maps to list of eval misclassification counts,
        // one scalar per tree
        std::unordered_map<int, std::vector<int>> acc;

        const int max_objs = budget - node->min_objective + 1;
        acc.reserve((size_t)std::max(1, max_objs));

        // leaf alternatives
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            int mistakes = 0;

            if (ctx.n_words > 0) {
                if (leaf.prediction == DEFER_PREDICTION) {
                    if (!BBwrong_eval) {
                        throw std::runtime_error(
                            "Deferred leaf encountered, but eval bb_pred was not provided."
                        );
                    }

                    mistakes = popcount_and_eval_(
                        eval_mask,
                        *BBwrong_eval,
                        ctx.n_words
                    );
                } else {
                    const int pred_class = leaf.prediction;
                    if (pred_class < 0 || pred_class >= num_classes) {
                        throw std::runtime_error("Leaf prediction is outside valid class range.");
                    }

                    const int n_here = count_eval_mask_(eval_mask, ctx.n_words);
                    const int correct = popcount_and_eval_(
                        eval_mask,
                        Y_eval_bits[(size_t)pred_class],
                        ctx.n_words
                    );

                    mistakes = n_here - correct;
                }
            }

            acc[leaf.loss].push_back(mistakes);
        }

        // split alternatives
        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;

            if (minL == INF || minR == INF) continue;

            int bL = budget - minR;
            int bR = budget - minL;

            if (bL < 0 || bR < 0) continue;

            // safety
            bL = std::min(bL, L->budget);
            bR = std::min(bR, R->budget);

            Packed Lmask((size_t)ctx.n_words);
            Packed Rmask((size_t)ctx.n_words);

            if (ctx.n_words > 0) {
                and_bits_eval(
                    eval_mask,
                    ctx.X_bits_eval[(size_t)split.feature],
                    Lmask,
                    ctx.n_words,
                    ctx.tail_mask
                );

                andnot_bits_eval(
                    eval_mask,
                    ctx.X_bits_eval[(size_t)split.feature],
                    Rmask,
                    ctx.n_words,
                    ctx.tail_mask
                );
            }

            auto Lb = collect_mistakes_by_obj_(L, bL, Lmask, ctx, Y_eval_bits, BBwrong_eval);
            auto Rb = collect_mistakes_by_obj_(R, bR, Rmask, ctx, Y_eval_bits, BBwrong_eval);

            if (Lb.empty() || Rb.empty()) continue;

            std::vector<int> R_objs;
            R_objs.reserve(Rb.size());
            for (const auto& rb : Rb) {
                R_objs.push_back(rb.obj);
            }

            for (const auto& lb : Lb) {
                const int lo = lb.obj;
                if (lo > budget) break;

                const int rem = budget - lo;

                auto it_end = std::upper_bound(R_objs.begin(), R_objs.end(), rem);
                const size_t r_hi = (size_t)std::distance(R_objs.begin(), it_end);

                if (r_hi == 0) continue;

                for (size_t ri = 0; ri < r_hi; ++ri) {
                    const int ro = Rb[ri].obj;
                    const int tot = lo + ro;

                    const auto& Lmis = lb.mistakes;
                    const auto& Rmis = Rb[ri].mistakes;

                    auto& dest = acc[tot];                    

                    for (int lm : Lmis) {
                        for (int rm : Rmis) {
                            dest.push_back(lm + rm);
                        }
                    }
                }
            }
        }

        return to_sorted_mistake_buckets_(acc);
    }


    struct ExactReplacementState_ {
        // these masks are only materialized after the replacement variable
        // has appeared on the current tree path. Until then, perturbing the
        // variable cannot change routing, so its leaf contribution is just
        // the ordinary/original leaf mistakes.
        Packed target_rows;
        Packed replacement_values;
        bool replacement_feature_used = false;
    };

    struct ExactReplacementCounts_ {
        int64_t original_mistakes = 0;
        std::vector<double> replacement_expected_mistakes;
    };

    struct ObjExactReplacementBucket_ {
        int obj = 0;
        std::vector<ExactReplacementCounts_> counts;
    };

    // reused by every recursive call. stamps let us avoid clearing O(G)
    // counters for every variable/leaf; we only reset groups actually touched
    // by the current target/donor masks.
    struct ExactMatchedScratch_ {
        std::vector<int> wrong_counts;
        std::vector<int> replacement_counts;
        std::vector<uint32_t> stamps;
        std::vector<int> touched_groups;
        uint32_t current_stamp = 0;

        inline void begin(int number_of_groups) {
            if (number_of_groups < 0) {
                throw std::runtime_error(
                    "Negative number of matched groups."
                );
            }

            const std::size_t need =
                static_cast<std::size_t>(number_of_groups);

            if (wrong_counts.size() < need) {
                wrong_counts.resize(need, 0);
                replacement_counts.resize(need, 0);
                stamps.resize(need, 0);
            }

            ++current_stamp;

            if (current_stamp == 0) {
                std::fill(stamps.begin(), stamps.end(), 0);
                current_stamp = 1;
            }

            touched_groups.clear();
        }

        inline void touch(int group) {
            const std::size_t g =
                static_cast<std::size_t>(group);

            if (stamps[g] == current_stamp) {
                return;
            }

            stamps[g] = current_stamp;
            wrong_counts[g] = 0;
            replacement_counts[g] = 0;
            touched_groups.push_back(group);
        }
    };

    static inline std::vector<ObjExactReplacementBucket_>
    to_sorted_exact_replacement_buckets_(
        std::unordered_map<
            int,
            std::vector<ExactReplacementCounts_>
        >& acc
    ) {
        std::vector<ObjExactReplacementBucket_> out;
        out.reserve(acc.size());

        for (auto& kv : acc) {
            ObjExactReplacementBucket_ b;
            b.obj = kv.first;
            b.counts = std::move(kv.second);
            out.push_back(std::move(b));
        }

        std::sort(
            out.begin(),
            out.end(),
            [](const ObjExactReplacementBucket_& a,
               const ObjExactReplacementBucket_& b) {
                return a.obj < b.obj;
            }
        );

        return out;
    }

    inline int exact_wrong_count_for_leaf_(
        const Packed& target_rows,
        int prediction,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval
    ) const {
        if (ctx.n_words <= 0) return 0;

        if (prediction == DEFER_PREDICTION) {
            if (!BBwrong_eval) {
                throw std::runtime_error(
                    "Deferred leaf encountered, but eval bb_pred was not provided."
                );
            }

            return popcount_and_eval_(
                target_rows,
                *BBwrong_eval,
                ctx.n_words
            );
        }

        if (prediction < 0 || prediction >= num_classes) {
            throw std::runtime_error(
                "Leaf prediction is outside valid class range."
            );
        }

        const int n_here =
            count_eval_mask_(target_rows, ctx.n_words);

        const int correct =
            popcount_and_eval_(
                target_rows,
                Y_eval_bits[(size_t)prediction],
                ctx.n_words
            );

        return n_here - correct;
    }

    static inline int exact_ctz64_(uint64_t bits) {
    #if defined(_MSC_VER)
        unsigned long bit_index = 0;
        _BitScanForward64(&bit_index, bits);
        return static_cast<int>(bit_index);
    #else
        return __builtin_ctzll(bits);
    #endif
    }

    std::vector<ObjExactReplacementBucket_>
    collect_exact_replacement_mistakes_by_obj_(
        const TreeTrieNode* node,
        int budget,

        // ordinary, unmodified evaluation rows at this graph node.
        const Packed& original_mask,

        // full evaluation-row mask. This is the donor universe before the
        // replacement variable appears on the path.
        const Packed& replacement_root_mask,

        // one state per original variable.
        const std::vector<ExactReplacementState_>& states,

        // internal binary-column index -> original variable index.
        const std::vector<int>& internal_to_variable,

        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,

        // matched_group_of_row_by_variable_eval[j][i]
        // gives the matched-group ID of evaluation row i
        // for replacement variable j.
        const std::vector<std::vector<int>>*
            matched_group_of_row_by_variable_eval,

        // Precomputed 1 / |G_g|. zero for groups absent from this bootstrap.
        const std::vector<std::vector<double>>*
            matched_group_inv_size_by_variable_eval,

        // 1 iff exactly one matched group is nonempty for this variable in
        // this bootstrap, in which case the conditional kernel is identical
        // to ordinary uniform replacement.
        const std::vector<uint8_t>*
            matched_group_effectively_uniform_by_variable_eval,

        ExactMatchedScratch_* matched_scratch
    ) const {
        if (!node || budget < 0) return {};

        constexpr int INF = std::numeric_limits<int>::max();

        if (node->min_objective == INF ||
            node->min_objective > budget) {
            return {};
        }

        const int number_of_variables =
            static_cast<int>(states.size());

        std::unordered_map<
            int,
            std::vector<ExactReplacementCounts_>
        > acc;

        const int max_objs =
            budget - node->min_objective + 1;

        acc.reserve((size_t)std::max(1, max_objs));

        // leaf alternatives
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            ExactReplacementCounts_ here;

            here.original_mistakes =
                exact_wrong_count_for_leaf_(
                    original_mask,
                    leaf.prediction,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval
                );

            here.replacement_expected_mistakes.assign(
                (std::size_t)number_of_variables,
                static_cast<double>(here.original_mistakes)
            );

            if (
                leaf.prediction != DEFER_PREDICTION &&
                (leaf.prediction < 0 ||
                 leaf.prediction >= num_classes)
            ) {
                throw std::runtime_error(
                    "Leaf prediction is outside valid class range."
                );
            }

            if (
                leaf.prediction == DEFER_PREDICTION &&
                !BBwrong_eval
            ) {
                throw std::runtime_error(
                    "Deferred leaf encountered, but eval bb_pred was not provided."
                );
            }

            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {

                const auto& state =
                    states[(std::size_t)variable];

                // if the replacement variable has not appeared anywhere on
                // this root-to-leaf path, perturbing it cannot change routing.
                // its expected perturbed mistakes are exactly the ordinary mistakes at this leaf.
                if (!state.replacement_feature_used) {
                    continue;
                }

                const bool matched_effectively_uniform =
                    matched_group_effectively_uniform_by_variable_eval !=
                        nullptr &&
                    (*matched_group_effectively_uniform_by_variable_eval)[
                        (std::size_t)variable
                    ] != 0;

                // ordinary uniform replacement, or a conditional partition with exactly one nonempty group  (which is exactly permutation importance)
                if (
                    matched_group_of_row_by_variable_eval == nullptr ||
                    matched_effectively_uniform
                ) {
                    const int wrong_target_rows =
                        exact_wrong_count_for_leaf_(
                            state.target_rows,
                            leaf.prediction,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval
                        );

                    const int number_of_replacement_values =
                        count_eval_mask_(
                            state.replacement_values,
                            ctx.n_words
                        );

                    here.replacement_expected_mistakes[
                        (std::size_t)variable
                    ] =
                        (
                            (double)wrong_target_rows *
                            (double)number_of_replacement_values
                        )
                        / (double)ctx.n_eval;

                    continue;
                }

                const auto& group_of_row =
                    (*matched_group_of_row_by_variable_eval)[
                        (std::size_t)variable
                    ];

                const auto& inverse_group_sizes =
                    (*matched_group_inv_size_by_variable_eval)[
                        (std::size_t)variable
                    ];

                const int number_of_groups =
                    static_cast<int>(
                        inverse_group_sizes.size()
                    );

                if (!matched_scratch) {
                    throw std::runtime_error(
                        "Matched-group scratch space was not provided."
                    );
                }

                matched_scratch->begin(number_of_groups);

                // count donor rows by group by iterating only the set bits of the current replacement-value mask
                for (int wi = 0; wi < ctx.n_words; ++wi) {
                    uint64_t replacement_bits =
                        state.replacement_values.w[
                            (std::size_t)wi
                        ];

                    while (replacement_bits) {
                        const int bit =
                            exact_ctz64_(replacement_bits);

                        const int row =
                            (wi << 6) + bit;

                        replacement_bits &=
                            replacement_bits - 1;

                        const int group =
                            group_of_row[
                                (std::size_t)row
                            ];

                        matched_scratch->touch(group);

                        ++matched_scratch->replacement_counts[
                            (std::size_t)group
                        ];
                    }

                    // count only target rows that are wrong for this leaf
                    uint64_t wrong_bits =
                        state.target_rows.w[
                            (std::size_t)wi
                        ];

                    if (leaf.prediction == DEFER_PREDICTION) {
                        wrong_bits &=
                            BBwrong_eval->w[
                                (std::size_t)wi
                            ];
                    } else {
                        wrong_bits &=
                            ~Y_eval_bits[
                                (std::size_t)leaf.prediction
                            ].w[
                                (std::size_t)wi
                            ];
                    }

                    while (wrong_bits) {
                        const int bit =
                            exact_ctz64_(wrong_bits);

                        const int row =
                            (wi << 6) + bit;

                        wrong_bits &=
                            wrong_bits - 1;

                        const int group =
                            group_of_row[
                                (std::size_t)row
                            ];

                        matched_scratch->touch(group);

                        ++matched_scratch->wrong_counts[
                            (std::size_t)group
                        ];
                    }
                }

                double expected_mistakes = 0.0;

                // only groups touched by at least one target or donor mask need to be visited
                for (int group :
                     matched_scratch->touched_groups) {

                    expected_mistakes +=
                        (
                            (double)matched_scratch->wrong_counts[
                                (std::size_t)group
                            ] *
                            (double)matched_scratch->replacement_counts[
                                (std::size_t)group
                            ]
                        ) *
                        inverse_group_sizes[
                            (std::size_t)group
                        ];
                }

                here.replacement_expected_mistakes[
                    (std::size_t)variable
                ] = expected_mistakes;
            }

            acc[leaf.loss].push_back(std::move(here));
        }

        // split alternatives
        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();

            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;

            if (minL == INF || minR == INF) continue;

            int bL = budget - minR;
            int bR = budget - minL;

            if (bL < 0 || bR < 0) continue;

            bL = std::min(bL, L->budget);
            bR = std::min(bR, R->budget);

            if (bL < minL || bR < minR) continue;

            if (
                split.feature < 0 ||
                split.feature >= (int)internal_to_variable.size()
            ) {
                throw std::runtime_error(
                    "Exact replacement evaluation saw an invalid split feature."
                );
            }

            const int split_variable =
                internal_to_variable[(size_t)split.feature];

            if (
                split_variable < 0 ||
                split_variable >= number_of_variables
            ) {
                throw std::runtime_error(
                    "Split feature is not mapped to an original variable."
                );
            }

            const Packed& Xf =
                ctx.X_bits_eval[(size_t)split.feature];

            // normal/original evaluation
            Packed original_left((size_t)ctx.n_words);
            Packed original_right((size_t)ctx.n_words);

            and_bits_eval(
                original_mask,
                Xf,
                original_left,
                ctx.n_words,
                ctx.tail_mask
            );

            andnot_bits_eval(
                original_mask,
                Xf,
                original_right,
                ctx.n_words,
                ctx.tail_mask
            );

            std::vector<ExactReplacementState_> left_states(
                (size_t)number_of_variables
            );
            std::vector<ExactReplacementState_> right_states(
                (size_t)number_of_variables
            );

            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {

                const auto& cur =
                    states[(size_t)variable];

                auto& ls =
                    left_states[(size_t)variable];

                auto& rs =
                    right_states[(size_t)variable];

                // lazy state of bitvectors before variable j first appears on the path 
                if (!cur.replacement_feature_used) {
                    if (variable != split_variable) {
                        continue;
                    }

                    // first occurrence of this replacement variable.
                    ls.replacement_feature_used = true;
                    rs.replacement_feature_used = true;

                    ls.target_rows = original_mask;
                    rs.target_rows = original_mask;

                    ls.replacement_values =
                        Packed((size_t)ctx.n_words);

                    rs.replacement_values =
                        Packed((size_t)ctx.n_words);

                    and_bits_eval(
                        replacement_root_mask,
                        Xf,
                        ls.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    andnot_bits_eval(
                        replacement_root_mask,
                        Xf,
                        rs.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    continue;
                }

                ls.replacement_feature_used = true;
                rs.replacement_feature_used = true;

                ls.target_rows =
                    Packed((size_t)ctx.n_words);

                rs.target_rows =
                    Packed((size_t)ctx.n_words);

                ls.replacement_values =
                    Packed((size_t)ctx.n_words);

                rs.replacement_values =
                    Packed((size_t)ctx.n_words);

                if (variable == split_variable) {
                    // this is the variable being replaced. The target rows
                    // do not split on their original value. instead, the set
                    // of possible replacement values splits.
                    ls.target_rows.w =
                        cur.target_rows.w;

                    rs.target_rows.w =
                        cur.target_rows.w;

                    and_bits_eval(
                        cur.replacement_values,
                        Xf,
                        ls.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    andnot_bits_eval(
                        cur.replacement_values,
                        Xf,
                        rs.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                } else {
                    // some other variable is split normally on the target
                    // row. the possible values of the replaced variable are unchanged.
                    and_bits_eval(
                        cur.target_rows,
                        Xf,
                        ls.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    andnot_bits_eval(
                        cur.target_rows,
                        Xf,
                        rs.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    ls.replacement_values.w =
                        cur.replacement_values.w;

                    rs.replacement_values.w =
                        cur.replacement_values.w;
                }
            }

            auto Lb =
                collect_exact_replacement_mistakes_by_obj_(
                    L,
                    bL,
                    original_left,
                    replacement_root_mask,
                    left_states,
                    internal_to_variable,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    matched_group_of_row_by_variable_eval,
                    matched_group_inv_size_by_variable_eval,
                    matched_group_effectively_uniform_by_variable_eval,
                    matched_scratch
                );

            auto Rb =
                collect_exact_replacement_mistakes_by_obj_(
                    R,
                    bR,
                    original_right,
                    replacement_root_mask,
                    right_states,
                    internal_to_variable,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    matched_group_of_row_by_variable_eval,
                    matched_group_inv_size_by_variable_eval,
                    matched_group_effectively_uniform_by_variable_eval,
                    matched_scratch
                );

            if (Lb.empty() || Rb.empty()) continue;

            std::vector<int> R_objs;
            R_objs.reserve(Rb.size());

            for (const auto& rb : Rb) {
                R_objs.push_back(rb.obj);
            }

            for (const auto& lb : Lb) {
                if (lb.obj > bL) continue;

                const int rem = budget - lb.obj;

                auto it_end =
                    std::upper_bound(
                        R_objs.begin(),
                        R_objs.end(),
                        rem
                    );

                const size_t r_hi =
                    (size_t)std::distance(
                        R_objs.begin(),
                        it_end
                    );

                for (size_t ri = 0; ri < r_hi; ++ri) {
                    const auto& rb = Rb[ri];

                    const int total_obj =
                        lb.obj + rb.obj;

                    if (total_obj > budget) continue;

                    auto& dest = acc[total_obj];

                    for (const auto& lc : lb.counts) {
                        for (const auto& rc : rb.counts) {
                            ExactReplacementCounts_ combined;

                            combined.original_mistakes =
                                lc.original_mistakes +
                                rc.original_mistakes;

                            combined.replacement_expected_mistakes.resize(
                                (std::size_t)number_of_variables
                            );

                            for (int variable = 0;
                                 variable < number_of_variables;
                                 ++variable) {

                                combined.replacement_expected_mistakes[
                                    (std::size_t)variable
                                ] =
                                    lc.replacement_expected_mistakes[
                                        (std::size_t)variable
                                    ] +
                                    rc.replacement_expected_mistakes[
                                        (std::size_t)variable
                                    ];
                            }

                            dest.push_back(std::move(combined));
                        }
                    }
                }
            }
        }

        return to_sorted_exact_replacement_buckets_(acc);
    }

    struct ExactGlobalImportanceExtrema_ {
        std::vector<double> lower;
        std::vector<double> upper;
    };

    struct ObjExactGlobalImportanceExtremaBucket_ {
        int obj = 0;
        ExactGlobalImportanceExtrema_ extrema;
    };

    struct ExactLocalImportanceExtrema_ {
        std::vector<double> lower;
        std::vector<double> upper;
    };

    struct ObjExactLocalImportanceExtremaBucket_ {
        int obj = 0;
        ExactLocalImportanceExtrema_ extrema;
    };

    static inline void update_global_extrema_(
        ExactGlobalImportanceExtrema_& dst,
        const std::vector<double>& candidate_lower,
        const std::vector<double>& candidate_upper
    ) {
        if (dst.lower.empty()) {
            dst.lower = candidate_lower;
            dst.upper = candidate_upper;
            return;
        }

        if (
            dst.lower.size() != candidate_lower.size() ||
            dst.upper.size() != candidate_upper.size()
        ) {
            throw std::runtime_error(
                "Global exact-importance extrema have inconsistent sizes."
            );
        }

        for (std::size_t j = 0; j < dst.lower.size(); ++j) {
            dst.lower[j] = std::min(dst.lower[j], candidate_lower[j]);
            dst.upper[j] = std::max(dst.upper[j], candidate_upper[j]);
        }
    }

    static inline void update_local_extrema_(
        ExactLocalImportanceExtrema_& dst,
        const std::vector<double>& candidate_lower,
        const std::vector<double>& candidate_upper
    ) {
        if (dst.lower.empty()) {
            dst.lower = candidate_lower;
            dst.upper = candidate_upper;
            return;
        }

        if (
            dst.lower.size() != candidate_lower.size() ||
            dst.upper.size() != candidate_upper.size()
        ) {
            throw std::runtime_error(
                "Local exact-importance extrema have inconsistent sizes."
            );
        }

        for (std::size_t i = 0; i < dst.lower.size(); ++i) {
            dst.lower[i] = std::min(dst.lower[i], candidate_lower[i]);
            dst.upper[i] = std::max(dst.upper[i], candidate_upper[i]);
        }
    }

    static inline void update_global_extrema_from_sum_(
        ExactGlobalImportanceExtrema_& dst,
        const ExactGlobalImportanceExtrema_& left,
        const ExactGlobalImportanceExtrema_& right
    ) {
        if (left.lower.size() != right.lower.size() ||
            left.upper.size() != right.upper.size() ||
            left.lower.size() != left.upper.size()) {
            throw std::runtime_error(
                "Global exact-importance extrema have inconsistent sizes."
            );
        }

        const std::size_t n = left.lower.size();

        if (dst.lower.empty()) {
            dst.lower.resize(n);
            dst.upper.resize(n);

            for (std::size_t j = 0; j < n; ++j) {
                dst.lower[j] = left.lower[j] + right.lower[j];
                dst.upper[j] = left.upper[j] + right.upper[j];
            }
            return;
        }

        if (dst.lower.size() != n || dst.upper.size() != n) {
            throw std::runtime_error(
                "Global exact-importance extrema have inconsistent sizes."
            );
        }

        for (std::size_t j = 0; j < n; ++j) {
            dst.lower[j] = std::min(
                dst.lower[j],
                left.lower[j] + right.lower[j]
            );
            dst.upper[j] = std::max(
                dst.upper[j],
                left.upper[j] + right.upper[j]
            );
        }
    }

    static inline void update_local_extrema_from_sum_(
        ExactLocalImportanceExtrema_& dst,
        const ExactLocalImportanceExtrema_& left,
        const ExactLocalImportanceExtrema_& right
    ) {
        if (left.lower.size() != right.lower.size() ||
            left.upper.size() != right.upper.size() ||
            left.lower.size() != left.upper.size()) {
            throw std::runtime_error(
                "Local exact-importance extrema have inconsistent sizes."
            );
        }

        const std::size_t n = left.lower.size();

        if (dst.lower.empty()) {
            dst.lower.resize(n);
            dst.upper.resize(n);

            for (std::size_t i = 0; i < n; ++i) {
                dst.lower[i] = left.lower[i] + right.lower[i];
                dst.upper[i] = left.upper[i] + right.upper[i];
            }
            return;
        }

        if (dst.lower.size() != n || dst.upper.size() != n) {
            throw std::runtime_error(
                "Local exact-importance extrema have inconsistent sizes."
            );
        }

        for (std::size_t i = 0; i < n; ++i) {
            dst.lower[i] = std::min(
                dst.lower[i],
                left.lower[i] + right.lower[i]
            );
            dst.upper[i] = std::max(
                dst.upper[i],
                left.upper[i] + right.upper[i]
            );
        }
    }

    static inline std::vector<ObjExactGlobalImportanceExtremaBucket_>
    to_sorted_global_importance_extrema_buckets_(
        std::unordered_map<int, ExactGlobalImportanceExtrema_>& acc
    ) {
        std::vector<ObjExactGlobalImportanceExtremaBucket_> out;
        out.reserve(acc.size());

        for (auto& kv : acc) {
            ObjExactGlobalImportanceExtremaBucket_ b;
            b.obj = kv.first;
            b.extrema = std::move(kv.second);
            out.push_back(std::move(b));
        }

        std::sort(
            out.begin(),
            out.end(),
            [](const ObjExactGlobalImportanceExtremaBucket_& a,
               const ObjExactGlobalImportanceExtremaBucket_& b) {
                return a.obj < b.obj;
            }
        );

        return out;
    }

    static inline std::vector<ObjExactLocalImportanceExtremaBucket_>
    to_sorted_local_importance_extrema_buckets_(
        std::unordered_map<int, ExactLocalImportanceExtrema_>& acc
    ) {
        std::vector<ObjExactLocalImportanceExtremaBucket_> out;
        out.reserve(acc.size());

        for (auto& kv : acc) {
            ObjExactLocalImportanceExtremaBucket_ b;
            b.obj = kv.first;
            b.extrema = std::move(kv.second);
            out.push_back(std::move(b));
        }

        std::sort(
            out.begin(),
            out.end(),
            [](const ObjExactLocalImportanceExtremaBucket_& a,
               const ObjExactLocalImportanceExtremaBucket_& b) {
                return a.obj < b.obj;
            }
        );

        return out;
    }

    double exact_replacement_expected_mistakes_for_leaf_variable_(
        const ExactReplacementState_& state,
        int variable,
        int prediction,
        int original_mistakes,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<std::vector<int>>*
            matched_group_of_row_by_variable_eval,
        const std::vector<std::vector<double>>*
            matched_group_inv_size_by_variable_eval,
        const std::vector<uint8_t>*
            matched_group_effectively_uniform_by_variable_eval,
        ExactMatchedScratch_* matched_scratch
    ) const {
        if (!state.replacement_feature_used) {
            return static_cast<double>(original_mistakes);
        }

        const bool matched_effectively_uniform =
            matched_group_effectively_uniform_by_variable_eval != nullptr &&
            (*matched_group_effectively_uniform_by_variable_eval)[
                static_cast<std::size_t>(variable)
            ] != 0;

        if (
            matched_group_of_row_by_variable_eval == nullptr ||
            matched_effectively_uniform
        ) {
            const int wrong_target_rows =
                exact_wrong_count_for_leaf_(
                    state.target_rows,
                    prediction,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval
                );

            const int number_of_replacement_values =
                count_eval_mask_(
                    state.replacement_values,
                    ctx.n_words
                );

            return
                (
                    static_cast<double>(wrong_target_rows) *
                    static_cast<double>(number_of_replacement_values)
                ) /
                static_cast<double>(ctx.n_eval);
        }

        if (
            matched_group_inv_size_by_variable_eval == nullptr ||
            matched_scratch == nullptr
        ) {
            throw std::runtime_error(
                "Matched-group exact replacement state is incomplete."
            );
        }

        const auto& group_of_row =
            (*matched_group_of_row_by_variable_eval)[
                static_cast<std::size_t>(variable)
            ];

        const auto& inverse_group_sizes =
            (*matched_group_inv_size_by_variable_eval)[
                static_cast<std::size_t>(variable)
            ];

        const int number_of_groups =
            static_cast<int>(inverse_group_sizes.size());

        matched_scratch->begin(number_of_groups);

        for (int wi = 0; wi < ctx.n_words; ++wi) {
            uint64_t replacement_bits =
                state.replacement_values.w[static_cast<std::size_t>(wi)];

            while (replacement_bits) {
                const int bit = exact_ctz64_(replacement_bits);
                const int row = (wi << 6) + bit;
                replacement_bits &= replacement_bits - 1;

                const int group =
                    group_of_row[static_cast<std::size_t>(row)];

                matched_scratch->touch(group);
                ++matched_scratch->replacement_counts[
                    static_cast<std::size_t>(group)
                ];
            }

            uint64_t wrong_bits =
                state.target_rows.w[static_cast<std::size_t>(wi)];

            if (prediction == DEFER_PREDICTION) {
                wrong_bits &=
                    BBwrong_eval->w[static_cast<std::size_t>(wi)];
            } else {
                wrong_bits &=
                    ~Y_eval_bits[
                        static_cast<std::size_t>(prediction)
                    ].w[static_cast<std::size_t>(wi)];
            }

            while (wrong_bits) {
                const int bit = exact_ctz64_(wrong_bits);
                const int row = (wi << 6) + bit;
                wrong_bits &= wrong_bits - 1;

                const int group =
                    group_of_row[static_cast<std::size_t>(row)];

                matched_scratch->touch(group);
                ++matched_scratch->wrong_counts[
                    static_cast<std::size_t>(group)
                ];
            }
        }

        double expected_mistakes = 0.0;

        for (int group : matched_scratch->touched_groups) {
            expected_mistakes +=
                static_cast<double>(
                    matched_scratch->wrong_counts[
                        static_cast<std::size_t>(group)
                    ]
                ) *
                static_cast<double>(
                    matched_scratch->replacement_counts[
                        static_cast<std::size_t>(group)
                    ]
                ) *
                inverse_group_sizes[static_cast<std::size_t>(group)];
        }

        return expected_mistakes;
    }

    std::vector<double> exact_local_importance_for_leaf_variable_(
        const Packed& original_mask,
        const ExactReplacementState_& state,
        int prediction,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<int>* matched_group_of_row_eval,
        const std::vector<double>* matched_group_inv_size_eval,
        bool matched_effectively_uniform,
        ExactMatchedScratch_* matched_scratch
    ) const {
        std::vector<double> local(
            static_cast<std::size_t>(ctx.n_eval),
            0.0
        );

        // if j never appeared on this root-to-leaf path, replacement and
        // original routing agree exactly on this leaf, hence contribution 0.
        if (!state.replacement_feature_used) {
            return local;
        }

        if (
            prediction != DEFER_PREDICTION &&
            (prediction < 0 || prediction >= num_classes)
        ) {
            throw std::runtime_error(
                "Leaf prediction is outside valid class range."
            );
        }

        if (prediction == DEFER_PREDICTION && !BBwrong_eval) {
            throw std::runtime_error(
                "Deferred leaf encountered, but eval bb_pred was not provided."
            );
        }

        // subtract the original 0/1 loss contribution of this leaf.
        for (int wi = 0; wi < ctx.n_words; ++wi) {
            uint64_t wrong_original =
                original_mask.w[static_cast<std::size_t>(wi)];

            if (prediction == DEFER_PREDICTION) {
                wrong_original &=
                    BBwrong_eval->w[static_cast<std::size_t>(wi)];
            } else {
                wrong_original &=
                    ~Y_eval_bits[
                        static_cast<std::size_t>(prediction)
                    ].w[static_cast<std::size_t>(wi)];
            }

            while (wrong_original) {
                const int bit = exact_ctz64_(wrong_original);
                const int row = (wi << 6) + bit;
                wrong_original &= wrong_original - 1;
                local[static_cast<std::size_t>(row)] -= 1.0;
            }
        }

        // ordinary empirical permutation importance, or matched permutation
        // with a single nonempty group, has the same donor probability for every target row.
        if (
            matched_group_of_row_eval == nullptr ||
            matched_effectively_uniform
        ) {
            const int number_of_replacement_values =
                count_eval_mask_(
                    state.replacement_values,
                    ctx.n_words
                );

            const double donor_probability =
                static_cast<double>(number_of_replacement_values) /
                static_cast<double>(ctx.n_eval);

            if (donor_probability == 0.0) {
                return local;
            }

            for (int wi = 0; wi < ctx.n_words; ++wi) {
                uint64_t wrong_target =
                    state.target_rows.w[static_cast<std::size_t>(wi)];

                if (prediction == DEFER_PREDICTION) {
                    wrong_target &=
                        BBwrong_eval->w[static_cast<std::size_t>(wi)];
                } else {
                    wrong_target &=
                        ~Y_eval_bits[
                            static_cast<std::size_t>(prediction)
                        ].w[static_cast<std::size_t>(wi)];
                }

                while (wrong_target) {
                    const int bit = exact_ctz64_(wrong_target);
                    const int row = (wi << 6) + bit;
                    wrong_target &= wrong_target - 1;
                    local[static_cast<std::size_t>(row)] +=
                        donor_probability;
                }
            }

            return local;
        }

        if (
            matched_group_inv_size_eval == nullptr ||
            matched_scratch == nullptr
        ) {
            throw std::runtime_error(
                "Matched-group local exact replacement state is incomplete."
            );
        }

        const int number_of_groups =
            static_cast<int>(matched_group_inv_size_eval->size());

        matched_scratch->begin(number_of_groups);

        // count donor values reaching this leaf by matched group
        for (int wi = 0; wi < ctx.n_words; ++wi) {
            uint64_t replacement_bits =
                state.replacement_values.w[static_cast<std::size_t>(wi)];

            while (replacement_bits) {
                const int bit = exact_ctz64_(replacement_bits);
                const int row = (wi << 6) + bit;
                replacement_bits &= replacement_bits - 1;

                const int group =
                    (*matched_group_of_row_eval)[
                        static_cast<std::size_t>(row)
                    ];

                matched_scratch->touch(group);
                ++matched_scratch->replacement_counts[
                    static_cast<std::size_t>(group)
                ];
            }
        }

        // add the perturbed expected loss contribution row by row.
        for (int wi = 0; wi < ctx.n_words; ++wi) {
            uint64_t wrong_target =
                state.target_rows.w[static_cast<std::size_t>(wi)];

            if (prediction == DEFER_PREDICTION) {
                wrong_target &=
                    BBwrong_eval->w[static_cast<std::size_t>(wi)];
            } else {
                wrong_target &=
                    ~Y_eval_bits[
                        static_cast<std::size_t>(prediction)
                    ].w[static_cast<std::size_t>(wi)];
            }

            while (wrong_target) {
                const int bit = exact_ctz64_(wrong_target);
                const int row = (wi << 6) + bit;
                wrong_target &= wrong_target - 1;

                const int group =
                    (*matched_group_of_row_eval)[
                        static_cast<std::size_t>(row)
                    ];

                // a target group can have zero donors reaching this leaf.
                matched_scratch->touch(group);

                local[static_cast<std::size_t>(row)] +=
                    static_cast<double>(
                        matched_scratch->replacement_counts[
                            static_cast<std::size_t>(group)
                        ]
                    ) *
                    (*matched_group_inv_size_eval)[
                        static_cast<std::size_t>(group)
                    ];
            }
        }

        return local;
    }

    std::vector<ObjExactGlobalImportanceExtremaBucket_>
    collect_exact_global_importance_extrema_by_obj_(
        const TreeTrieNode* node,
        int budget,
        const Packed& original_mask,
        const Packed& replacement_root_mask,
        const std::vector<ExactReplacementState_>& states,
        const std::vector<int>& internal_to_variable,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<std::vector<int>>*
            matched_group_of_row_by_variable_eval,
        const std::vector<std::vector<double>>*
            matched_group_inv_size_by_variable_eval,
        const std::vector<uint8_t>*
            matched_group_effectively_uniform_by_variable_eval,
        ExactMatchedScratch_* matched_scratch
    ) const {
        if (!node || budget < 0) return {};

        constexpr int INF = std::numeric_limits<int>::max();

        if (
            node->min_objective == INF ||
            node->min_objective > budget
        ) {
            return {};
        }

        const int number_of_variables =
            static_cast<int>(states.size());

        std::unordered_map<int, ExactGlobalImportanceExtrema_> acc;
        acc.reserve(
            static_cast<std::size_t>(
                std::max(1, budget - node->min_objective + 1)
            )
        );

        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            const int original_mistakes =
                exact_wrong_count_for_leaf_(
                    original_mask,
                    leaf.prediction,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval
                );

            std::vector<double> here(
                static_cast<std::size_t>(number_of_variables),
                0.0
            );

            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {

                if (!states[static_cast<std::size_t>(variable)]
                         .replacement_feature_used) {
                    continue;
                }

                const double replacement_mistakes =
                    exact_replacement_expected_mistakes_for_leaf_variable_(
                        states[static_cast<std::size_t>(variable)],
                        variable,
                        leaf.prediction,
                        original_mistakes,
                        ctx,
                        Y_eval_bits,
                        BBwrong_eval,
                        matched_group_of_row_by_variable_eval,
                        matched_group_inv_size_by_variable_eval,
                        matched_group_effectively_uniform_by_variable_eval,
                        matched_scratch
                    );

                here[static_cast<std::size_t>(variable)] =
                    replacement_mistakes -
                    static_cast<double>(original_mistakes);
            }

            update_global_extrema_(
                acc[leaf.loss],
                here,
                here
            );
        }

        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();

            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;

            if (minL == INF || minR == INF) continue;

            int bL = budget - minR;
            int bR = budget - minL;

            if (bL < 0 || bR < 0) continue;

            bL = std::min(bL, L->budget);
            bR = std::min(bR, R->budget);

            if (bL < minL || bR < minR) continue;

            if (
                split.feature < 0 ||
                split.feature >=
                    static_cast<int>(internal_to_variable.size())
            ) {
                throw std::runtime_error(
                    "Exact interval evaluation saw an invalid split feature."
                );
            }

            const int split_variable =
                internal_to_variable[
                    static_cast<std::size_t>(split.feature)
                ];

            if (
                split_variable < 0 ||
                split_variable >= number_of_variables
            ) {
                throw std::runtime_error(
                    "Split feature is not mapped to an original variable."
                );
            }

            const Packed& Xf =
                ctx.X_bits_eval[
                    static_cast<std::size_t>(split.feature)
                ];

            Packed original_left(static_cast<std::size_t>(ctx.n_words));
            Packed original_right(static_cast<std::size_t>(ctx.n_words));

            and_bits_eval(
                original_mask,
                Xf,
                original_left,
                ctx.n_words,
                ctx.tail_mask
            );

            andnot_bits_eval(
                original_mask,
                Xf,
                original_right,
                ctx.n_words,
                ctx.tail_mask
            );

            std::vector<ExactReplacementState_> left_states(
                static_cast<std::size_t>(number_of_variables)
            );

            std::vector<ExactReplacementState_> right_states(
                static_cast<std::size_t>(number_of_variables)
            );

            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {

                const auto& cur =
                    states[static_cast<std::size_t>(variable)];

                auto& ls =
                    left_states[static_cast<std::size_t>(variable)];

                auto& rs =
                    right_states[static_cast<std::size_t>(variable)];

                if (!cur.replacement_feature_used) {
                    if (variable != split_variable) {
                        continue;
                    }

                    ls.replacement_feature_used = true;
                    rs.replacement_feature_used = true;
                    ls.target_rows = original_mask;
                    rs.target_rows = original_mask;
                    ls.replacement_values =
                        Packed(static_cast<std::size_t>(ctx.n_words));
                    rs.replacement_values =
                        Packed(static_cast<std::size_t>(ctx.n_words));

                    and_bits_eval(
                        replacement_root_mask,
                        Xf,
                        ls.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    andnot_bits_eval(
                        replacement_root_mask,
                        Xf,
                        rs.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    continue;
                }

                ls.replacement_feature_used = true;
                rs.replacement_feature_used = true;
                ls.target_rows = Packed(static_cast<std::size_t>(ctx.n_words));
                rs.target_rows = Packed(static_cast<std::size_t>(ctx.n_words));
                ls.replacement_values = Packed(static_cast<std::size_t>(ctx.n_words));
                rs.replacement_values = Packed(static_cast<std::size_t>(ctx.n_words));

                if (variable == split_variable) {
                    ls.target_rows.w = cur.target_rows.w;
                    rs.target_rows.w = cur.target_rows.w;

                    and_bits_eval(
                        cur.replacement_values,
                        Xf,
                        ls.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    andnot_bits_eval(
                        cur.replacement_values,
                        Xf,
                        rs.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                } else {
                    and_bits_eval(
                        cur.target_rows,
                        Xf,
                        ls.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    andnot_bits_eval(
                        cur.target_rows,
                        Xf,
                        rs.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    ls.replacement_values.w = cur.replacement_values.w;
                    rs.replacement_values.w = cur.replacement_values.w;
                }
            }

            auto Lb =
                collect_exact_global_importance_extrema_by_obj_(
                    L,
                    bL,
                    original_left,
                    replacement_root_mask,
                    left_states,
                    internal_to_variable,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    matched_group_of_row_by_variable_eval,
                    matched_group_inv_size_by_variable_eval,
                    matched_group_effectively_uniform_by_variable_eval,
                    matched_scratch
                );

            auto Rb =
                collect_exact_global_importance_extrema_by_obj_(
                    R,
                    bR,
                    original_right,
                    replacement_root_mask,
                    right_states,
                    internal_to_variable,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    matched_group_of_row_by_variable_eval,
                    matched_group_inv_size_by_variable_eval,
                    matched_group_effectively_uniform_by_variable_eval,
                    matched_scratch
                );

            if (Lb.empty() || Rb.empty()) continue;

            std::vector<int> R_objs;
            R_objs.reserve(Rb.size());
            for (const auto& rb : Rb) {
                R_objs.push_back(rb.obj);
            }

            for (const auto& lb : Lb) {
                const int rem = budget - lb.obj;

                auto it_end =
                    std::upper_bound(
                        R_objs.begin(),
                        R_objs.end(),
                        rem
                    );

                const std::size_t r_hi =
                    static_cast<std::size_t>(
                        std::distance(R_objs.begin(), it_end)
                    );

                for (std::size_t ri = 0; ri < r_hi; ++ri) {
                    const auto& rb = Rb[ri];
                    const int total_obj = lb.obj + rb.obj;
                    if (total_obj > budget) continue;

                    update_global_extrema_from_sum_(
                        acc[total_obj],
                        lb.extrema,
                        rb.extrema
                    );
                }
            }
        }

        return to_sorted_global_importance_extrema_buckets_(acc);
    }

    std::vector<ObjExactLocalImportanceExtremaBucket_>
    collect_exact_local_importance_extrema_by_obj_(
        const TreeTrieNode* node,
        int budget,
        const Packed& original_mask,
        const Packed& replacement_root_mask,
        const ExactReplacementState_& state,
        int variable,
        const std::vector<int>& internal_to_variable,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<int>* matched_group_of_row_eval,
        const std::vector<double>* matched_group_inv_size_eval,
        bool matched_effectively_uniform,
        ExactMatchedScratch_* matched_scratch
    ) const {
        if (!node || budget < 0) return {};

        constexpr int INF = std::numeric_limits<int>::max();

        if (
            node->min_objective == INF ||
            node->min_objective > budget
        ) {
            return {};
        }

        std::unordered_map<int, ExactLocalImportanceExtrema_> acc;
        acc.reserve(
            static_cast<std::size_t>(
                std::max(1, budget - node->min_objective + 1)
            )
        );

        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            auto here =
                exact_local_importance_for_leaf_variable_(
                    original_mask,
                    state,
                    leaf.prediction,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    matched_group_of_row_eval,
                    matched_group_inv_size_eval,
                    matched_effectively_uniform,
                    matched_scratch
                );

            update_local_extrema_(
                acc[leaf.loss],
                here,
                here
            );
        }

        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();

            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;

            if (minL == INF || minR == INF) continue;

            int bL = budget - minR;
            int bR = budget - minL;

            if (bL < 0 || bR < 0) continue;

            bL = std::min(bL, L->budget);
            bR = std::min(bR, R->budget);

            if (bL < minL || bR < minR) continue;

            if (
                split.feature < 0 ||
                split.feature >=
                    static_cast<int>(internal_to_variable.size())
            ) {
                throw std::runtime_error(
                    "Exact local interval evaluation saw an invalid split feature."
                );
            }

            const int split_variable =
                internal_to_variable[
                    static_cast<std::size_t>(split.feature)
                ];

            if (split_variable < 0) {
                throw std::runtime_error(
                    "Split feature is not mapped to an original variable."
                );
            }

            const Packed& Xf =
                ctx.X_bits_eval[
                    static_cast<std::size_t>(split.feature)
                ];

            Packed original_left(static_cast<std::size_t>(ctx.n_words));
            Packed original_right(static_cast<std::size_t>(ctx.n_words));

            and_bits_eval(
                original_mask,
                Xf,
                original_left,
                ctx.n_words,
                ctx.tail_mask
            );

            andnot_bits_eval(
                original_mask,
                Xf,
                original_right,
                ctx.n_words,
                ctx.tail_mask
            );

            ExactReplacementState_ left_state;
            ExactReplacementState_ right_state;

            if (!state.replacement_feature_used) {
                if (split_variable == variable) {
                    left_state.replacement_feature_used = true;
                    right_state.replacement_feature_used = true;
                    left_state.target_rows = original_mask;
                    right_state.target_rows = original_mask;
                    left_state.replacement_values =
                        Packed(static_cast<std::size_t>(ctx.n_words));
                    right_state.replacement_values =
                        Packed(static_cast<std::size_t>(ctx.n_words));

                    and_bits_eval(
                        replacement_root_mask,
                        Xf,
                        left_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    andnot_bits_eval(
                        replacement_root_mask,
                        Xf,
                        right_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                }
            } else {
                left_state.replacement_feature_used = true;
                right_state.replacement_feature_used = true;
                left_state.target_rows =
                    Packed(static_cast<std::size_t>(ctx.n_words));
                right_state.target_rows =
                    Packed(static_cast<std::size_t>(ctx.n_words));
                left_state.replacement_values =
                    Packed(static_cast<std::size_t>(ctx.n_words));
                right_state.replacement_values =
                    Packed(static_cast<std::size_t>(ctx.n_words));

                if (split_variable == variable) {
                    left_state.target_rows.w = state.target_rows.w;
                    right_state.target_rows.w = state.target_rows.w;

                    and_bits_eval(
                        state.replacement_values,
                        Xf,
                        left_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    andnot_bits_eval(
                        state.replacement_values,
                        Xf,
                        right_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                } else {
                    and_bits_eval(
                        state.target_rows,
                        Xf,
                        left_state.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    andnot_bits_eval(
                        state.target_rows,
                        Xf,
                        right_state.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );

                    left_state.replacement_values.w =
                        state.replacement_values.w;
                    right_state.replacement_values.w =
                        state.replacement_values.w;
                }
            }

            auto Lb =
                collect_exact_local_importance_extrema_by_obj_(
                    L,
                    bL,
                    original_left,
                    replacement_root_mask,
                    left_state,
                    variable,
                    internal_to_variable,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    matched_group_of_row_eval,
                    matched_group_inv_size_eval,
                    matched_effectively_uniform,
                    matched_scratch
                );

            auto Rb =
                collect_exact_local_importance_extrema_by_obj_(
                    R,
                    bR,
                    original_right,
                    replacement_root_mask,
                    right_state,
                    variable,
                    internal_to_variable,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    matched_group_of_row_eval,
                    matched_group_inv_size_eval,
                    matched_effectively_uniform,
                    matched_scratch
                );

            if (Lb.empty() || Rb.empty()) continue;

            std::vector<int> R_objs;
            R_objs.reserve(Rb.size());
            for (const auto& rb : Rb) {
                R_objs.push_back(rb.obj);
            }

            for (const auto& lb : Lb) {
                const int rem = budget - lb.obj;

                auto it_end =
                    std::upper_bound(
                        R_objs.begin(),
                        R_objs.end(),
                        rem
                    );

                const std::size_t r_hi =
                    static_cast<std::size_t>(
                        std::distance(R_objs.begin(), it_end)
                    );

                for (std::size_t ri = 0; ri < r_hi; ++ri) {
                    const auto& rb = Rb[ri];
                    const int total_obj = lb.obj + rb.obj;
                    if (total_obj > budget) continue;

                    update_local_extrema_from_sum_(
                        acc[total_obj],
                        lb.extrema,
                        rb.extrema
                    );
                }
            }
        }

        return to_sorted_local_importance_extrema_buckets_(acc);
    }

    struct ExactAllLocalImportanceExtrema_ {
        int number_of_variables = 0;
        int n_eval = 0;
        std::vector<double> lower; // flattened [variable * n_eval + row]
        std::vector<double> upper;

        inline bool empty() const {
            return lower.empty();
        }

        inline std::size_t block_size() const {
            return static_cast<std::size_t>(n_eval);
        }

        inline double* lower_ptr(int variable) {
            return lower.data() +
                static_cast<std::size_t>(variable) * block_size();
        }

        inline double* upper_ptr(int variable) {
            return upper.data() +
                static_cast<std::size_t>(variable) * block_size();
        }

        inline const double* lower_ptr(int variable) const {
            return lower.data() +
                static_cast<std::size_t>(variable) * block_size();
        }

        inline const double* upper_ptr(int variable) const {
            return upper.data() +
                static_cast<std::size_t>(variable) * block_size();
        }
    };

    static inline void fast_update_minmax_(
        double* dst_lower,
        double* dst_upper,
        const double* cand_lower,
        const double* cand_upper,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        for (; i + 8 <= n; i += 8) {
            const __m512d dlo = _mm512_loadu_pd(dst_lower + i);
            const __m512d dup = _mm512_loadu_pd(dst_upper + i);
            const __m512d clo = _mm512_loadu_pd(cand_lower + i);
            const __m512d cup = _mm512_loadu_pd(cand_upper + i);
            _mm512_storeu_pd(dst_lower + i, _mm512_min_pd(dlo, clo));
            _mm512_storeu_pd(dst_upper + i, _mm512_max_pd(dup, cup));
        }
    #endif
        for (; i < n; ++i) {
            dst_lower[i] = std::min(dst_lower[i], cand_lower[i]);
            dst_upper[i] = std::max(dst_upper[i], cand_upper[i]);
        }
    }

    static inline void fast_update_minmax_sum_(
        double* dst_lower,
        double* dst_upper,
        const double* left_lower,
        const double* left_upper,
        const double* right_lower,
        const double* right_upper,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        for (; i + 8 <= n; i += 8) {
            const __m512d dlo = _mm512_loadu_pd(dst_lower + i);
            const __m512d dup = _mm512_loadu_pd(dst_upper + i);
            const __m512d llo = _mm512_loadu_pd(left_lower + i);
            const __m512d lup = _mm512_loadu_pd(left_upper + i);
            const __m512d rlo = _mm512_loadu_pd(right_lower + i);
            const __m512d rup = _mm512_loadu_pd(right_upper + i);
            const __m512d clo = _mm512_add_pd(llo, rlo);
            const __m512d cup = _mm512_add_pd(lup, rup);
            _mm512_storeu_pd(dst_lower + i, _mm512_min_pd(dlo, clo));
            _mm512_storeu_pd(dst_upper + i, _mm512_max_pd(dup, cup));
        }
    #endif
        for (; i < n; ++i) {
            const double lo = left_lower[i] + right_lower[i];
            const double up = left_upper[i] + right_upper[i];
            dst_lower[i] = std::min(dst_lower[i], lo);
            dst_upper[i] = std::max(dst_upper[i], up);
        }
    }

    static inline void fast_include_zero_(
        double* dst_lower,
        double* dst_upper,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        const __m512d z = _mm512_setzero_pd();
        for (; i + 8 <= n; i += 8) {
            const __m512d dlo = _mm512_loadu_pd(dst_lower + i);
            const __m512d dup = _mm512_loadu_pd(dst_upper + i);
            _mm512_storeu_pd(dst_lower + i, _mm512_min_pd(dlo, z));
            _mm512_storeu_pd(dst_upper + i, _mm512_max_pd(dup, z));
        }
    #endif
        for (; i < n; ++i) {
            dst_lower[i] = std::min(dst_lower[i], 0.0);
            dst_upper[i] = std::max(dst_upper[i], 0.0);
        }
    }

    static inline double fast_sum_doubles_(
        const double* x,
        std::size_t n
    ) {
        std::size_t i = 0;
        double total = 0.0;
    #if ArborEnum_USE_AVX512
        __m512d acc = _mm512_setzero_pd();
        for (; i + 8 <= n; i += 8) {
            acc = _mm512_add_pd(acc, _mm512_loadu_pd(x + i));
        }
        alignas(64) double tmp[8];
        _mm512_store_pd(tmp, acc);
        total = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                tmp[4] + tmp[5] + tmp[6] + tmp[7];
    #endif
        for (; i < n; ++i) total += x[i];
        return total;
    }

    void exact_local_importance_for_leaf_variable_into_(
        const Packed& original_mask,
        const ExactReplacementState_& state,
        int prediction,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<int>* matched_group_of_row_eval,
        const std::vector<double>* matched_group_inv_size_eval,
        bool matched_effectively_uniform,
        ExactMatchedScratch_* matched_scratch,
        std::vector<double>& local
    ) const {
        local.assign(static_cast<std::size_t>(ctx.n_eval), 0.0);

        if (!state.replacement_feature_used) {
            return;
        }

        if (
            prediction != DEFER_PREDICTION &&
            (prediction < 0 || prediction >= num_classes)
        ) {
            throw std::runtime_error(
                "Leaf prediction is outside valid class range."
            );
        }

        if (prediction == DEFER_PREDICTION && !BBwrong_eval) {
            throw std::runtime_error(
                "Deferred leaf encountered, but eval bb_pred was not provided."
            );
        }

        // Original contribution: -1 for rows misclassified by the original route.
        for (int wi = 0; wi < ctx.n_words; ++wi) {
            uint64_t wrong_original =
                original_mask.w[static_cast<std::size_t>(wi)];

            if (prediction == DEFER_PREDICTION) {
                wrong_original &=
                    BBwrong_eval->w[static_cast<std::size_t>(wi)];
            } else {
                wrong_original &=
                    ~Y_eval_bits[
                        static_cast<std::size_t>(prediction)
                    ].w[static_cast<std::size_t>(wi)];
            }

            while (wrong_original) {
                const int bit = exact_ctz64_(wrong_original);
                const int row = (wi << 6) + bit;
                wrong_original &= wrong_original - 1;
                local[static_cast<std::size_t>(row)] -= 1.0;
            }
        }

        if (
            matched_group_of_row_eval == nullptr ||
            matched_effectively_uniform
        ) {
            const int number_of_replacement_values =
                count_eval_mask_(state.replacement_values, ctx.n_words);

            const double donor_probability =
                static_cast<double>(number_of_replacement_values) /
                static_cast<double>(ctx.n_eval);

            if (donor_probability == 0.0) {
                return;
            }

            for (int wi = 0; wi < ctx.n_words; ++wi) {
                uint64_t wrong_target =
                    state.target_rows.w[static_cast<std::size_t>(wi)];

                if (prediction == DEFER_PREDICTION) {
                    wrong_target &=
                        BBwrong_eval->w[static_cast<std::size_t>(wi)];
                } else {
                    wrong_target &=
                        ~Y_eval_bits[
                            static_cast<std::size_t>(prediction)
                        ].w[static_cast<std::size_t>(wi)];
                }

                while (wrong_target) {
                    const int bit = exact_ctz64_(wrong_target);
                    const int row = (wi << 6) + bit;
                    wrong_target &= wrong_target - 1;
                    local[static_cast<std::size_t>(row)] += donor_probability;
                }
            }

            return;
        }

        if (
            matched_group_inv_size_eval == nullptr ||
            matched_scratch == nullptr
        ) {
            throw std::runtime_error(
                "Matched-group local exact replacement state is incomplete."
            );
        }

        const int number_of_groups =
            static_cast<int>(matched_group_inv_size_eval->size());

        matched_scratch->begin(number_of_groups);

        for (int wi = 0; wi < ctx.n_words; ++wi) {
            uint64_t replacement_bits =
                state.replacement_values.w[static_cast<std::size_t>(wi)];

            while (replacement_bits) {
                const int bit = exact_ctz64_(replacement_bits);
                const int row = (wi << 6) + bit;
                replacement_bits &= replacement_bits - 1;

                const int group =
                    (*matched_group_of_row_eval)[
                        static_cast<std::size_t>(row)
                    ];

                matched_scratch->touch(group);
                ++matched_scratch->replacement_counts[
                    static_cast<std::size_t>(group)
                ];
            }
        }

        for (int wi = 0; wi < ctx.n_words; ++wi) {
            uint64_t wrong_target =
                state.target_rows.w[static_cast<std::size_t>(wi)];

            if (prediction == DEFER_PREDICTION) {
                wrong_target &=
                    BBwrong_eval->w[static_cast<std::size_t>(wi)];
            } else {
                wrong_target &=
                    ~Y_eval_bits[
                        static_cast<std::size_t>(prediction)
                    ].w[static_cast<std::size_t>(wi)];
            }

            while (wrong_target) {
                const int bit = exact_ctz64_(wrong_target);
                const int row = (wi << 6) + bit;
                wrong_target &= wrong_target - 1;

                const int group =
                    (*matched_group_of_row_eval)[
                        static_cast<std::size_t>(row)
                    ];

                matched_scratch->touch(group);
                local[static_cast<std::size_t>(row)] +=
                    static_cast<double>(
                        matched_scratch->replacement_counts[
                            static_cast<std::size_t>(group)
                        ]
                    ) *
                    (*matched_group_inv_size_eval)[
                        static_cast<std::size_t>(group)
                    ];
            }
        }
    }


    struct ExactLocalImportanceNumeratorExtrema_ {
        std::vector<int32_t> point;
        std::vector<int32_t> lower;
        std::vector<int32_t> upper;
        bool all_zero = false;

        inline bool empty() const {
            return !all_zero && point.empty() && lower.empty();
        }

        inline bool is_point() const {
            return !point.empty();
        }
    };

    using ExactNodeVariableMasks_ =
        std::unordered_map<const TreeTrieNode*, std::vector<uint64_t>>;

    static inline void fast_update_minmax_i32_(
        int32_t* dst_lower,
        int32_t* dst_upper,
        const int32_t* cand_lower,
        const int32_t* cand_upper,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        for (; i + 16 <= n; i += 16) {
            const __m512i dlo = _mm512_loadu_si512((const void*)(dst_lower + i));
            const __m512i dup = _mm512_loadu_si512((const void*)(dst_upper + i));
            const __m512i clo = _mm512_loadu_si512((const void*)(cand_lower + i));
            const __m512i cup = _mm512_loadu_si512((const void*)(cand_upper + i));
            _mm512_storeu_si512((void*)(dst_lower + i), _mm512_min_epi32(dlo, clo));
            _mm512_storeu_si512((void*)(dst_upper + i), _mm512_max_epi32(dup, cup));
        }
    #endif
        for (; i < n; ++i) {
            dst_lower[i] = std::min(dst_lower[i], cand_lower[i]);
            dst_upper[i] = std::max(dst_upper[i], cand_upper[i]);
        }
    }

    static inline void fast_update_minmax_point_i32_(
        int32_t* dst_lower,
        int32_t* dst_upper,
        const int32_t* cand,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        for (; i + 16 <= n; i += 16) {
            const __m512i dlo = _mm512_loadu_si512((const void*)(dst_lower + i));
            const __m512i dup = _mm512_loadu_si512((const void*)(dst_upper + i));
            const __m512i c = _mm512_loadu_si512((const void*)(cand + i));
            _mm512_storeu_si512((void*)(dst_lower + i), _mm512_min_epi32(dlo, c));
            _mm512_storeu_si512((void*)(dst_upper + i), _mm512_max_epi32(dup, c));
        }
    #endif
        for (; i < n; ++i) {
            dst_lower[i] = std::min(dst_lower[i], cand[i]);
            dst_upper[i] = std::max(dst_upper[i], cand[i]);
        }
    }

    static inline void fast_update_minmax_sum_i32_(
        int32_t* dst_lower,
        int32_t* dst_upper,
        const int32_t* left_lower,
        const int32_t* left_upper,
        const int32_t* right_lower,
        const int32_t* right_upper,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        for (; i + 16 <= n; i += 16) {
            const __m512i dlo = _mm512_loadu_si512((const void*)(dst_lower + i));
            const __m512i dup = _mm512_loadu_si512((const void*)(dst_upper + i));
            const __m512i llo = _mm512_loadu_si512((const void*)(left_lower + i));
            const __m512i lup = _mm512_loadu_si512((const void*)(left_upper + i));
            const __m512i rlo = _mm512_loadu_si512((const void*)(right_lower + i));
            const __m512i rup = _mm512_loadu_si512((const void*)(right_upper + i));
            const __m512i clo = _mm512_add_epi32(llo, rlo);
            const __m512i cup = _mm512_add_epi32(lup, rup);
            _mm512_storeu_si512((void*)(dst_lower + i), _mm512_min_epi32(dlo, clo));
            _mm512_storeu_si512((void*)(dst_upper + i), _mm512_max_epi32(dup, cup));
        }
    #endif
        for (; i < n; ++i) {
            const int32_t lo = left_lower[i] + right_lower[i];
            const int32_t up = left_upper[i] + right_upper[i];
            dst_lower[i] = std::min(dst_lower[i], lo);
            dst_upper[i] = std::max(dst_upper[i], up);
        }
    }

    static inline void fast_update_minmax_point_sum_i32_(
        int32_t* dst_lower,
        int32_t* dst_upper,
        const int32_t* left,
        const int32_t* right,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        for (; i + 16 <= n; i += 16) {
            const __m512i dlo = _mm512_loadu_si512((const void*)(dst_lower + i));
            const __m512i dup = _mm512_loadu_si512((const void*)(dst_upper + i));
            const __m512i l = _mm512_loadu_si512((const void*)(left + i));
            const __m512i r = _mm512_loadu_si512((const void*)(right + i));
            const __m512i c = _mm512_add_epi32(l, r);
            _mm512_storeu_si512((void*)(dst_lower + i), _mm512_min_epi32(dlo, c));
            _mm512_storeu_si512((void*)(dst_upper + i), _mm512_max_epi32(dup, c));
        }
    #endif
        for (; i < n; ++i) {
            const int32_t c = left[i] + right[i];
            dst_lower[i] = std::min(dst_lower[i], c);
            dst_upper[i] = std::max(dst_upper[i], c);
        }
    }

    static inline void fast_update_minmax_point_interval_sum_i32_(
        int32_t* dst_lower,
        int32_t* dst_upper,
        const int32_t* point,
        const int32_t* int_lower,
        const int32_t* int_upper,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        for (; i + 16 <= n; i += 16) {
            const __m512i dlo = _mm512_loadu_si512((const void*)(dst_lower + i));
            const __m512i dup = _mm512_loadu_si512((const void*)(dst_upper + i));
            const __m512i p = _mm512_loadu_si512((const void*)(point + i));
            const __m512i lo = _mm512_add_epi32(p, _mm512_loadu_si512((const void*)(int_lower + i)));
            const __m512i up = _mm512_add_epi32(p, _mm512_loadu_si512((const void*)(int_upper + i)));
            _mm512_storeu_si512((void*)(dst_lower + i), _mm512_min_epi32(dlo, lo));
            _mm512_storeu_si512((void*)(dst_upper + i), _mm512_max_epi32(dup, up));
        }
    #endif
        for (; i < n; ++i) {
            const int32_t lo = point[i] + int_lower[i];
            const int32_t up = point[i] + int_upper[i];
            dst_lower[i] = std::min(dst_lower[i], lo);
            dst_upper[i] = std::max(dst_upper[i], up);
        }
    }

    static inline void fast_include_zero_i32_(
        int32_t* dst_lower,
        int32_t* dst_upper,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        const __m512i z = _mm512_setzero_si512();
        for (; i + 16 <= n; i += 16) {
            const __m512i dlo = _mm512_loadu_si512((const void*)(dst_lower + i));
            const __m512i dup = _mm512_loadu_si512((const void*)(dst_upper + i));
            _mm512_storeu_si512((void*)(dst_lower + i), _mm512_min_epi32(dlo, z));
            _mm512_storeu_si512((void*)(dst_upper + i), _mm512_max_epi32(dup, z));
        }
    #endif
        for (; i < n; ++i) {
            dst_lower[i] = std::min<int32_t>(dst_lower[i], 0);
            dst_upper[i] = std::max<int32_t>(dst_upper[i], 0);
        }
    }

    static inline void fast_add_inplace_i32_(
        int32_t* dst,
        const int32_t* src,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        for (; i + 16 <= n; i += 16) {
            const __m512i a = _mm512_loadu_si512((const void*)(dst + i));
            const __m512i b = _mm512_loadu_si512((const void*)(src + i));
            _mm512_storeu_si512((void*)(dst + i), _mm512_add_epi32(a, b));
        }
    #endif
        for (; i < n; ++i) dst[i] += src[i];
    }

    static inline void fast_add_point_to_interval_inplace_i32_(
        int32_t* lower,
        int32_t* upper,
        const int32_t* point,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        for (; i + 16 <= n; i += 16) {
            const __m512i p = _mm512_loadu_si512((const void*)(point + i));
            const __m512i lo = _mm512_loadu_si512((const void*)(lower + i));
            const __m512i up = _mm512_loadu_si512((const void*)(upper + i));
            _mm512_storeu_si512((void*)(lower + i), _mm512_add_epi32(lo, p));
            _mm512_storeu_si512((void*)(upper + i), _mm512_add_epi32(up, p));
        }
    #endif
        for (; i < n; ++i) {
            lower[i] += point[i];
            upper[i] += point[i];
        }
    }

    static inline void fast_add_interval_inplace_i32_(
        int32_t* dst_lower,
        int32_t* dst_upper,
        const int32_t* src_lower,
        const int32_t* src_upper,
        std::size_t n
    ) {
        std::size_t i = 0;
    #if ArborEnum_USE_AVX512
        for (; i + 16 <= n; i += 16) {
            const __m512i dlo = _mm512_loadu_si512((const void*)(dst_lower + i));
            const __m512i dup = _mm512_loadu_si512((const void*)(dst_upper + i));
            const __m512i slo = _mm512_loadu_si512((const void*)(src_lower + i));
            const __m512i sup = _mm512_loadu_si512((const void*)(src_upper + i));
            _mm512_storeu_si512((void*)(dst_lower + i), _mm512_add_epi32(dlo, slo));
            _mm512_storeu_si512((void*)(dst_upper + i), _mm512_add_epi32(dup, sup));
        }
    #endif
        for (; i < n; ++i) {
            dst_lower[i] += src_lower[i];
            dst_upper[i] += src_upper[i];
        }
    }

    static inline int64_t fast_sum_i32_(
        const int32_t* x,
        std::size_t n
    ) {
        int64_t total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            total += static_cast<int64_t>(x[i]);
        }
        return total;
    }

    static inline void promote_point_numerator_extrema_(
        ExactLocalImportanceNumeratorExtrema_& x
    ) {
        if (!x.is_point()) return;
        x.upper = x.point;
        x.lower = std::move(x.point);
    }

    static inline void materialize_zero_numerator_extrema_(
        ExactLocalImportanceNumeratorExtrema_& x,
        std::size_t n
    ) {
        if (!x.all_zero) return;
        x.lower.assign(n, 0);
        x.upper.assign(n, 0);
        x.all_zero = false;
    }

    static inline void merge_numerator_extrema_(
        ExactLocalImportanceNumeratorExtrema_& dst,
        const ExactLocalImportanceNumeratorExtrema_& cand,
        std::size_t n
    ) {
        if (cand.empty()) return;

        if (dst.empty()) {
            dst = cand;
            return;
        }

        if (cand.all_zero) {
            if (dst.all_zero) return;
            if (dst.is_point()) promote_point_numerator_extrema_(dst);
            fast_include_zero_i32_(dst.lower.data(), dst.upper.data(), n);
            return;
        }

        if (dst.all_zero) {
            dst.all_zero = false;
            if (cand.is_point()) {
                dst.lower = cand.point;
                dst.upper = cand.point;
            } else {
                dst.lower = cand.lower;
                dst.upper = cand.upper;
            }
            fast_include_zero_i32_(dst.lower.data(), dst.upper.data(), n);
            return;
        }

        if (dst.is_point()) promote_point_numerator_extrema_(dst);

        if (cand.is_point()) {
            fast_update_minmax_point_i32_(
                dst.lower.data(), dst.upper.data(), cand.point.data(), n
            );
        } else {
            fast_update_minmax_i32_(
                dst.lower.data(), dst.upper.data(),
                cand.lower.data(), cand.upper.data(), n
            );
        }
    }

    static inline void merge_numerator_extrema_move_(
        ExactLocalImportanceNumeratorExtrema_& dst,
        ExactLocalImportanceNumeratorExtrema_&& cand,
        std::size_t n
    ) {
        if (cand.empty()) return;
        if (dst.empty()) {
            dst = std::move(cand);
            return;
        }

        if (cand.all_zero) {
            if (dst.all_zero) return;
            if (dst.is_point()) promote_point_numerator_extrema_(dst);
            fast_include_zero_i32_(dst.lower.data(), dst.upper.data(), n);
            return;
        }

        if (dst.all_zero) {
            dst.all_zero = false;
            if (cand.is_point()) {
                dst.lower = std::move(cand.point);
                dst.upper = dst.lower;
            } else {
                dst.lower = std::move(cand.lower);
                dst.upper = std::move(cand.upper);
            }
            fast_include_zero_i32_(dst.lower.data(), dst.upper.data(), n);
            return;
        }

        if (dst.is_point()) promote_point_numerator_extrema_(dst);

        if (cand.is_point()) {
            fast_update_minmax_point_i32_(
                dst.lower.data(), dst.upper.data(), cand.point.data(), n
            );
        } else {
            fast_update_minmax_i32_(
                dst.lower.data(), dst.upper.data(),
                cand.lower.data(), cand.upper.data(), n
            );
        }
    }

    static inline void update_numerator_extrema_from_sum_(
        ExactLocalImportanceNumeratorExtrema_& dst,
        ExactLocalImportanceNumeratorExtrema_&& left,
        ExactLocalImportanceNumeratorExtrema_&& right,
        std::size_t n
    ) {
        if (left.empty() || right.empty()) return;

        if (left.all_zero && right.all_zero) {
            ExactLocalImportanceNumeratorExtrema_ z;
            z.all_zero = true;
            merge_numerator_extrema_move_(dst, std::move(z), n);
            return;
        }
        if (left.all_zero) {
            merge_numerator_extrema_move_(dst, std::move(right), n);
            return;
        }
        if (right.all_zero) {
            merge_numerator_extrema_move_(dst, std::move(left), n);
            return;
        }

        const bool lp = left.is_point();
        const bool rp = right.is_point();

        if (dst.empty()) {
            if (lp && rp) {
                fast_add_inplace_i32_(left.point.data(), right.point.data(), n);
                dst = std::move(left);
            } else if (lp && !rp) {
                fast_add_point_to_interval_inplace_i32_(
                    right.lower.data(), right.upper.data(), left.point.data(), n
                );
                dst = std::move(right);
            } else if (!lp && rp) {
                fast_add_point_to_interval_inplace_i32_(
                    left.lower.data(), left.upper.data(), right.point.data(), n
                );
                dst = std::move(left);
            } else {
                fast_add_interval_inplace_i32_(
                    left.lower.data(), left.upper.data(),
                    right.lower.data(), right.upper.data(), n
                );
                dst = std::move(left);
            }
            return;
        }

        if (dst.all_zero) materialize_zero_numerator_extrema_(dst, n);
        if (dst.is_point()) promote_point_numerator_extrema_(dst);

        if (lp && rp) {
            fast_update_minmax_point_sum_i32_(
                dst.lower.data(), dst.upper.data(),
                left.point.data(), right.point.data(), n
            );
        } else if (lp && !rp) {
            fast_update_minmax_point_interval_sum_i32_(
                dst.lower.data(), dst.upper.data(),
                left.point.data(), right.lower.data(), right.upper.data(), n
            );
        } else if (!lp && rp) {
            fast_update_minmax_point_interval_sum_i32_(
                dst.lower.data(), dst.upper.data(),
                right.point.data(), left.lower.data(), left.upper.data(), n
            );
        } else {
            fast_update_minmax_sum_i32_(
                dst.lower.data(), dst.upper.data(),
                left.lower.data(), left.upper.data(),
                right.lower.data(), right.upper.data(), n
            );
        }
    }

    void build_exact_node_variable_masks_(
        const TreeTrieNode* node,
        const std::vector<int>& internal_to_variable,
        int number_of_variables,
        ExactNodeVariableMasks_& masks
    ) const {
        if (!node || masks.find(node) != masks.end()) return;

        const std::size_t word_count =
            static_cast<std::size_t>((number_of_variables + 63) >> 6);
        std::vector<uint64_t> mask(word_count, 0ULL);

        for (const auto& split : node->splits) {
            const int variable =
                internal_to_variable[
                    static_cast<std::size_t>(split.feature)
                ];
            mask[static_cast<std::size_t>(variable >> 6)] |=
                1ULL << (variable & 63);

            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();

            build_exact_node_variable_masks_(
                L,
                internal_to_variable,
                number_of_variables,
                masks
            );
            build_exact_node_variable_masks_(
                R,
                internal_to_variable,
                number_of_variables,
                masks
            );

            if (L) {
                auto it = masks.find(L);
                if (it != masks.end()) {
                    for (std::size_t w = 0; w < word_count; ++w) {
                        mask[w] |= it->second[w];
                    }
                }
            }

            if (R) {
                auto it = masks.find(R);
                if (it != masks.end()) {
                    for (std::size_t w = 0; w < word_count; ++w) {
                        mask[w] |= it->second[w];
                    }
                }
            }
        }

        masks.emplace(node, std::move(mask));
    }

    static inline bool exact_node_can_contain_variable_(
        const TreeTrieNode* node,
        int variable,
        const ExactNodeVariableMasks_& masks
    ) {
        const auto it = masks.find(node);
        if (it == masks.end()) return false;
        const std::size_t w = static_cast<std::size_t>(variable >> 6);
        return
            w < it->second.size() &&
            ((it->second[w] >> (variable & 63)) & 1ULL) != 0ULL;
    }

    void update_numerator_extrema_from_leaf_(
        ExactLocalImportanceNumeratorExtrema_& acc,
        const Packed& original_mask,
        const ExactReplacementState_& state,
        int prediction,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<int>* matched_group_of_row_eval,
        const std::vector<int>* matched_group_size_eval,
        bool matched_effectively_uniform,
        ExactMatchedScratch_* matched_scratch
    ) const {
        const std::size_t n = static_cast<std::size_t>(ctx.n_eval);

        if (!state.replacement_feature_used) {
            ExactLocalImportanceNumeratorExtrema_ z;
            z.all_zero = true;
            merge_numerator_extrema_move_(acc, std::move(z), n);
            return;
        }

        if (
            prediction != DEFER_PREDICTION &&
            (prediction < 0 || prediction >= num_classes)
        ) {
            throw std::runtime_error(
                "Leaf prediction is outside valid class range."
            );
        }

        if (prediction == DEFER_PREDICTION && !BBwrong_eval) {
            throw std::runtime_error(
                "Deferred leaf encountered, but eval bb_pred was not provided."
            );
        }

        const bool grouped =
            matched_group_of_row_eval != nullptr &&
            !matched_effectively_uniform;

        int donor_count = 0;

        if (!grouped) {
            donor_count =
                count_eval_mask_(state.replacement_values, ctx.n_words);
        } else {
            if (
                matched_group_size_eval == nullptr ||
                matched_scratch == nullptr
            ) {
                throw std::runtime_error(
                    "Matched-group local exact replacement state is incomplete."
                );
            }

            const int number_of_groups =
                static_cast<int>(matched_group_size_eval->size());
            matched_scratch->begin(number_of_groups);

            for (int wi = 0; wi < ctx.n_words; ++wi) {
                uint64_t replacement_bits =
                    state.replacement_values.w[
                        static_cast<std::size_t>(wi)
                    ];

                while (replacement_bits) {
                    const int bit = exact_ctz64_(replacement_bits);
                    const int row = (wi << 6) + bit;
                    replacement_bits &= replacement_bits - 1;
                    const int group =
                        (*matched_group_of_row_eval)[
                            static_cast<std::size_t>(row)
                        ];
                    matched_scratch->touch(group);
                    ++matched_scratch->replacement_counts[
                        static_cast<std::size_t>(group)
                    ];
                }
            }
        }

        auto wrong_word = [&](const Packed& mask, int wi) -> uint64_t {
            uint64_t bits = mask.w[static_cast<std::size_t>(wi)];
            if (prediction == DEFER_PREDICTION) {
                bits &= BBwrong_eval->w[static_cast<std::size_t>(wi)];
            } else {
                bits &=
                    ~Y_eval_bits[
                        static_cast<std::size_t>(prediction)
                    ].w[static_cast<std::size_t>(wi)];
            }
            if (wi == ctx.n_words - 1) bits &= ctx.tail_mask;
            return bits;
        };

        bool any_special = false;
        for (int wi = 0; wi < ctx.n_words; ++wi) {
            if (
                (wrong_word(original_mask, wi) |
                 wrong_word(state.target_rows, wi)) != 0ULL
            ) {
                any_special = true;
                break;
            }
        }

        if (!any_special) {
            ExactLocalImportanceNumeratorExtrema_ z;
            z.all_zero = true;
            merge_numerator_extrema_move_(acc, std::move(z), n);
            return;
        }

        const bool was_empty = acc.empty();
        const bool had_zero = acc.all_zero;

        if (was_empty) {
            acc.point.assign(n, 0);
            acc.all_zero = false;

            for (int wi = 0; wi < ctx.n_words; ++wi) {
                const uint64_t wo = wrong_word(original_mask, wi);
                const uint64_t wt = wrong_word(state.target_rows, wi);
                uint64_t special = wo | wt;

                while (special) {
                    const int bit = exact_ctz64_(special);
                    const int row = (wi << 6) + bit;
                    special &= special - 1;

                    int32_t value = 0;

                    if (grouped) {
                        const int group =
                            (*matched_group_of_row_eval)[
                                static_cast<std::size_t>(row)
                            ];
                        if ((wt >> bit) & 1ULL) {
                            matched_scratch->touch(group);
                            value +=
                                matched_scratch->replacement_counts[
                                    static_cast<std::size_t>(group)
                                ];
                        }
                        if ((wo >> bit) & 1ULL) {
                            value -=
                                (*matched_group_size_eval)[
                                    static_cast<std::size_t>(group)
                                ];
                        }
                    } else {
                        if ((wt >> bit) & 1ULL) value += donor_count;
                        if ((wo >> bit) & 1ULL) value -= ctx.n_eval;
                    }

                    acc.point[static_cast<std::size_t>(row)] = value;
                }
            }
            return;
        }

        if (had_zero) {
            acc.lower.assign(n, 0);
            acc.upper.assign(n, 0);
            acc.all_zero = false;
        } else if (acc.is_point()) {
            promote_point_numerator_extrema_(acc);
        }

        for (int wi = 0; wi < ctx.n_words; ++wi) {
            const uint64_t wo = wrong_word(original_mask, wi);
            const uint64_t wt = wrong_word(state.target_rows, wi);
            const uint64_t special = wo | wt;
            const int base = wi << 6;
            const int count = std::min(64, ctx.n_eval - base);

            if (special == 0ULL) {
                fast_include_zero_i32_(
                    acc.lower.data() + static_cast<std::size_t>(base),
                    acc.upper.data() + static_cast<std::size_t>(base),
                    static_cast<std::size_t>(count)
                );
                continue;
            }

            for (int bit = 0; bit < count; ++bit) {
                int32_t value = 0;

                if ((special >> bit) & 1ULL) {
                    if (grouped) {
                        const int row = base + bit;
                        const int group =
                            (*matched_group_of_row_eval)[
                                static_cast<std::size_t>(row)
                            ];
                        if ((wt >> bit) & 1ULL) {
                            matched_scratch->touch(group);
                            value +=
                                matched_scratch->replacement_counts[
                                    static_cast<std::size_t>(group)
                                ];
                        }
                        if ((wo >> bit) & 1ULL) {
                            value -=
                                (*matched_group_size_eval)[
                                    static_cast<std::size_t>(group)
                                ];
                        }
                    } else {
                        if ((wt >> bit) & 1ULL) value += donor_count;
                        if ((wo >> bit) & 1ULL) value -= ctx.n_eval;
                    }
                }

                const std::size_t row =
                    static_cast<std::size_t>(base + bit);
                acc.lower[row] = std::min(acc.lower[row], value);
                acc.upper[row] = std::max(acc.upper[row], value);
            }
        }
    }

    ExactLocalImportanceNumeratorExtrema_
    collect_exact_local_importance_numerator_extrema_at_most_(
        const TreeTrieNode* node,
        int budget,
        const Packed& original_mask,
        const Packed& replacement_root_mask,
        const ExactReplacementState_& state,
        int variable,
        const std::vector<int>& internal_to_variable,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<int>* matched_group_of_row_eval,
        const std::vector<int>* matched_group_size_eval,
        bool matched_effectively_uniform,
        ExactMatchedScratch_* matched_scratch,
        const ExactNodeVariableMasks_& node_variable_masks
    ) const {
        ExactLocalImportanceNumeratorExtrema_ acc;

        if (!node || budget < 0) return acc;

        constexpr int INF = std::numeric_limits<int>::max();

        if (
            node->min_objective == INF ||
            node->min_objective > budget
        ) {
            return acc;
        }

        if (
            !state.replacement_feature_used &&
            !exact_node_can_contain_variable_(
                node,
                variable,
                node_variable_masks
            )
        ) {
            acc.all_zero = true;
            return acc;
        }

        const std::size_t n = static_cast<std::size_t>(ctx.n_eval);

        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            update_numerator_extrema_from_leaf_(
                acc,
                original_mask,
                state,
                leaf.prediction,
                ctx,
                Y_eval_bits,
                BBwrong_eval,
                matched_group_of_row_eval,
                matched_group_size_eval,
                matched_effectively_uniform,
                matched_scratch
            );
        }

        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;
            if (minL == INF || minR == INF) continue;
            if (minL + minR > budget) continue;

            if (
                split.feature < 0 ||
                split.feature >=
                    static_cast<int>(internal_to_variable.size())
            ) {
                throw std::runtime_error(
                    "Exact local interval evaluation saw an invalid split feature."
                );
            }

            const int split_variable =
                internal_to_variable[
                    static_cast<std::size_t>(split.feature)
                ];

            if (split_variable < 0) {
                throw std::runtime_error(
                    "Split feature is not mapped to an original variable."
                );
            }

            const Packed& Xf =
                ctx.X_bits_eval[
                    static_cast<std::size_t>(split.feature)
                ];

            Packed original_left(static_cast<std::size_t>(ctx.n_words));
            Packed original_right(static_cast<std::size_t>(ctx.n_words));

            and_bits_eval(
                original_mask,
                Xf,
                original_left,
                ctx.n_words,
                ctx.tail_mask
            );
            andnot_bits_eval(
                original_mask,
                Xf,
                original_right,
                ctx.n_words,
                ctx.tail_mask
            );

            ExactReplacementState_ left_state;
            ExactReplacementState_ right_state;

            if (!state.replacement_feature_used) {
                if (split_variable == variable) {
                    left_state.replacement_feature_used = true;
                    right_state.replacement_feature_used = true;
                    left_state.target_rows = original_mask;
                    right_state.target_rows = original_mask;
                    left_state.replacement_values =
                        Packed(static_cast<std::size_t>(ctx.n_words));
                    right_state.replacement_values =
                        Packed(static_cast<std::size_t>(ctx.n_words));

                    and_bits_eval(
                        replacement_root_mask,
                        Xf,
                        left_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        replacement_root_mask,
                        Xf,
                        right_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                }
            } else {
                left_state.replacement_feature_used = true;
                right_state.replacement_feature_used = true;
                left_state.target_rows =
                    Packed(static_cast<std::size_t>(ctx.n_words));
                right_state.target_rows =
                    Packed(static_cast<std::size_t>(ctx.n_words));
                left_state.replacement_values =
                    Packed(static_cast<std::size_t>(ctx.n_words));
                right_state.replacement_values =
                    Packed(static_cast<std::size_t>(ctx.n_words));

                if (split_variable == variable) {
                    left_state.target_rows.w = state.target_rows.w;
                    right_state.target_rows.w = state.target_rows.w;

                    and_bits_eval(
                        state.replacement_values,
                        Xf,
                        left_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        state.replacement_values,
                        Xf,
                        right_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                } else {
                    and_bits_eval(
                        state.target_rows,
                        Xf,
                        left_state.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        state.target_rows,
                        Xf,
                        right_state.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    left_state.replacement_values.w =
                        state.replacement_values.w;
                    right_state.replacement_values.w =
                        state.replacement_values.w;
                }
            }

            if (split_variable != variable) {
                const int left_budget =
                    std::min(L->budget, budget - minR);
                const int right_budget =
                    std::min(R->budget, budget - minL);

                auto left_ext =
                    collect_exact_local_importance_numerator_extrema_at_most_(
                        L,
                        left_budget,
                        original_left,
                        replacement_root_mask,
                        left_state,
                        variable,
                        internal_to_variable,
                        ctx,
                        Y_eval_bits,
                        BBwrong_eval,
                        matched_group_of_row_eval,
                        matched_group_size_eval,
                        matched_effectively_uniform,
                        matched_scratch,
                        node_variable_masks
                    );

                auto right_ext =
                    collect_exact_local_importance_numerator_extrema_at_most_(
                        R,
                        right_budget,
                        original_right,
                        replacement_root_mask,
                        right_state,
                        variable,
                        internal_to_variable,
                        ctx,
                        Y_eval_bits,
                        BBwrong_eval,
                        matched_group_of_row_eval,
                        matched_group_size_eval,
                        matched_effectively_uniform,
                        matched_scratch,
                        node_variable_masks
                    );

                if (left_ext.empty() || right_ext.empty()) continue;

                update_numerator_extrema_from_sum_(
                    acc,
                    std::move(left_ext),
                    std::move(right_ext),
                    n
                );
                continue;
            }

            L->ensure_hist_built();
            R->ensure_hist_built();

            const int max_left_budget =
                std::min(L->budget, budget - minR);
            const int max_right_budget =
                std::min(R->budget, budget - minL);

            std::size_t left_count = 0;
            for (const auto& e : L->hist) {
                if (e.obj > max_left_budget) break;
                ++left_count;
            }

            std::size_t right_count = 0;
            for (const auto& e : R->hist) {
                if (e.obj > max_right_budget) break;
                ++right_count;
            }

            ExactLocalImportanceNumeratorExtrema_ split_acc;

            if (left_count <= right_count) {
                for (const auto& e : L->hist) {
                    if (e.obj > max_left_budget) break;

                    const int left_budget = e.obj;
                    const int right_budget =
                        std::min(R->budget, budget - left_budget);
                    if (right_budget < minR) continue;

                    auto left_ext =
                        collect_exact_local_importance_numerator_extrema_at_most_(
                            L,
                            left_budget,
                            original_left,
                            replacement_root_mask,
                            left_state,
                            variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            matched_group_of_row_eval,
                            matched_group_size_eval,
                            matched_effectively_uniform,
                            matched_scratch,
                            node_variable_masks
                        );

                    auto right_ext =
                        collect_exact_local_importance_numerator_extrema_at_most_(
                            R,
                            right_budget,
                            original_right,
                            replacement_root_mask,
                            right_state,
                            variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            matched_group_of_row_eval,
                            matched_group_size_eval,
                            matched_effectively_uniform,
                            matched_scratch,
                            node_variable_masks
                        );

                    if (left_ext.empty() || right_ext.empty()) continue;

                    update_numerator_extrema_from_sum_(
                        split_acc,
                        std::move(left_ext),
                        std::move(right_ext),
                        n
                    );
                }
            } else {
                for (const auto& e : R->hist) {
                    if (e.obj > max_right_budget) break;

                    const int right_budget = e.obj;
                    const int left_budget =
                        std::min(L->budget, budget - right_budget);
                    if (left_budget < minL) continue;

                    auto left_ext =
                        collect_exact_local_importance_numerator_extrema_at_most_(
                            L,
                            left_budget,
                            original_left,
                            replacement_root_mask,
                            left_state,
                            variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            matched_group_of_row_eval,
                            matched_group_size_eval,
                            matched_effectively_uniform,
                            matched_scratch,
                            node_variable_masks
                        );

                    auto right_ext =
                        collect_exact_local_importance_numerator_extrema_at_most_(
                            R,
                            right_budget,
                            original_right,
                            replacement_root_mask,
                            right_state,
                            variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            matched_group_of_row_eval,
                            matched_group_size_eval,
                            matched_effectively_uniform,
                            matched_scratch,
                            node_variable_masks
                        );

                    if (left_ext.empty() || right_ext.empty()) continue;

                    update_numerator_extrema_from_sum_(
                        split_acc,
                        std::move(left_ext),
                        std::move(right_ext),
                        n
                    );
                }
            }

            if (!split_acc.empty()) {
                merge_numerator_extrema_move_(
                    acc,
                    std::move(split_acc),
                    n
                );
            }
        }

        return acc;
    }


    std::vector<ExactLocalImportanceNumeratorExtrema_>
    collect_exact_local_importance_numerator_extrema_all_at_most_(
        const TreeTrieNode* node,
        int budget,
        const Packed& original_mask,
        const Packed& replacement_root_mask,
        const std::vector<ExactReplacementState_>& states,
        const std::vector<uint8_t>& requested,
        const std::vector<int>& internal_to_variable,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<std::vector<int>>* matched_group_of_row_by_variable_eval,
        const std::vector<std::vector<int>>* matched_group_size_by_variable_eval,
        const std::vector<uint8_t>* matched_group_effectively_uniform_by_variable_eval,
        ExactMatchedScratch_* matched_scratch,
        const ExactNodeVariableMasks_& node_variable_masks
    ) const {
        if (!node || budget < 0) return {};

        constexpr int INF = std::numeric_limits<int>::max();
        if (node->min_objective == INF || node->min_objective > budget) {
            return {};
        }

        const int number_of_variables = static_cast<int>(states.size());
        std::vector<ExactLocalImportanceNumeratorExtrema_> acc(
            static_cast<std::size_t>(number_of_variables)
        );
        std::vector<uint8_t> active = requested;
        bool any_active = false;

        for (int variable = 0; variable < number_of_variables; ++variable) {
            const std::size_t j = static_cast<std::size_t>(variable);
            if (!active[j]) continue;
            if (
                !states[j].replacement_feature_used &&
                !exact_node_can_contain_variable_(
                    node,
                    variable,
                    node_variable_masks
                )
            ) {
                acc[j].all_zero = true;
                active[j] = 0;
            } else {
                any_active = true;
            }
        }

        if (!any_active) return acc;

        const std::size_t n = static_cast<std::size_t>(ctx.n_eval);

        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            for (int variable = 0; variable < number_of_variables; ++variable) {
                const std::size_t j = static_cast<std::size_t>(variable);
                if (!active[j]) continue;

                const std::vector<int>* group_of_row_ptr =
                    matched_group_of_row_by_variable_eval
                        ? &(*matched_group_of_row_by_variable_eval)[j]
                        : nullptr;
                const std::vector<int>* group_size_ptr =
                    matched_group_size_by_variable_eval
                        ? &(*matched_group_size_by_variable_eval)[j]
                        : nullptr;
                const bool matched_effectively_uniform =
                    matched_group_effectively_uniform_by_variable_eval &&
                    (*matched_group_effectively_uniform_by_variable_eval)[j] != 0;

                update_numerator_extrema_from_leaf_(
                    acc[j],
                    original_mask,
                    states[j],
                    leaf.prediction,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    group_of_row_ptr,
                    group_size_ptr,
                    matched_effectively_uniform,
                    matched_scratch
                );
            }
        }

        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;
            if (minL == INF || minR == INF) continue;
            if (minL + minR > budget) continue;

            if (
                split.feature < 0 ||
                split.feature >= static_cast<int>(internal_to_variable.size())
            ) {
                throw std::runtime_error(
                    "Exact local all-variable interval evaluation saw an invalid split feature."
                );
            }

            const int split_variable =
                internal_to_variable[static_cast<std::size_t>(split.feature)];
            if (split_variable < 0 || split_variable >= number_of_variables) {
                throw std::runtime_error(
                    "Split feature is not mapped to an original variable."
                );
            }

            const Packed& Xf =
                ctx.X_bits_eval[static_cast<std::size_t>(split.feature)];

            Packed original_left(static_cast<std::size_t>(ctx.n_words));
            Packed original_right(static_cast<std::size_t>(ctx.n_words));
            and_bits_eval(
                original_mask,
                Xf,
                original_left,
                ctx.n_words,
                ctx.tail_mask
            );
            andnot_bits_eval(
                original_mask,
                Xf,
                original_right,
                ctx.n_words,
                ctx.tail_mask
            );

            std::vector<ExactReplacementState_> left_states(
                static_cast<std::size_t>(number_of_variables)
            );
            std::vector<ExactReplacementState_> right_states(
                static_cast<std::size_t>(number_of_variables)
            );

            for (int variable = 0; variable < number_of_variables; ++variable) {
                const std::size_t j = static_cast<std::size_t>(variable);
                if (!active[j]) continue;

                const auto& cur = states[j];
                auto& ls = left_states[j];
                auto& rs = right_states[j];

                if (!cur.replacement_feature_used) {
                    if (variable == split_variable) {
                        ls.replacement_feature_used = true;
                        rs.replacement_feature_used = true;
                        ls.target_rows = original_mask;
                        rs.target_rows = original_mask;
                        ls.replacement_values = Packed(
                            static_cast<std::size_t>(ctx.n_words)
                        );
                        rs.replacement_values = Packed(
                            static_cast<std::size_t>(ctx.n_words)
                        );
                        and_bits_eval(
                            replacement_root_mask,
                            Xf,
                            ls.replacement_values,
                            ctx.n_words,
                            ctx.tail_mask
                        );
                        andnot_bits_eval(
                            replacement_root_mask,
                            Xf,
                            rs.replacement_values,
                            ctx.n_words,
                            ctx.tail_mask
                        );
                    }
                    continue;
                }

                ls.replacement_feature_used = true;
                rs.replacement_feature_used = true;
                ls.target_rows = Packed(static_cast<std::size_t>(ctx.n_words));
                rs.target_rows = Packed(static_cast<std::size_t>(ctx.n_words));
                ls.replacement_values = Packed(static_cast<std::size_t>(ctx.n_words));
                rs.replacement_values = Packed(static_cast<std::size_t>(ctx.n_words));

                if (variable == split_variable) {
                    ls.target_rows.w = cur.target_rows.w;
                    rs.target_rows.w = cur.target_rows.w;
                    and_bits_eval(
                        cur.replacement_values,
                        Xf,
                        ls.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        cur.replacement_values,
                        Xf,
                        rs.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                } else {
                    and_bits_eval(
                        cur.target_rows,
                        Xf,
                        ls.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        cur.target_rows,
                        Xf,
                        rs.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    ls.replacement_values.w = cur.replacement_values.w;
                    rs.replacement_values.w = cur.replacement_values.w;
                }
            }

            const int left_budget = std::min(L->budget, budget - minR);
            const int right_budget = std::min(R->budget, budget - minL);

            std::vector<uint8_t> shared_requested = active;
            const std::size_t split_j = static_cast<std::size_t>(split_variable);
            const bool do_split_variable = shared_requested[split_j] != 0;
            shared_requested[split_j] = 0;

            bool any_shared = false;
            for (uint8_t x : shared_requested) {
                if (x) {
                    any_shared = true;
                    break;
                }
            }

            if (any_shared) {
                auto left_all =
                    collect_exact_local_importance_numerator_extrema_all_at_most_(
                        L,
                        left_budget,
                        original_left,
                        replacement_root_mask,
                        left_states,
                        shared_requested,
                        internal_to_variable,
                        ctx,
                        Y_eval_bits,
                        BBwrong_eval,
                        matched_group_of_row_by_variable_eval,
                        matched_group_size_by_variable_eval,
                        matched_group_effectively_uniform_by_variable_eval,
                        matched_scratch,
                        node_variable_masks
                    );
                auto right_all =
                    collect_exact_local_importance_numerator_extrema_all_at_most_(
                        R,
                        right_budget,
                        original_right,
                        replacement_root_mask,
                        right_states,
                        shared_requested,
                        internal_to_variable,
                        ctx,
                        Y_eval_bits,
                        BBwrong_eval,
                        matched_group_of_row_by_variable_eval,
                        matched_group_size_by_variable_eval,
                        matched_group_effectively_uniform_by_variable_eval,
                        matched_scratch,
                        node_variable_masks
                    );

                if (!left_all.empty() && !right_all.empty()) {
                    for (int variable = 0; variable < number_of_variables; ++variable) {
                        const std::size_t j = static_cast<std::size_t>(variable);
                        if (!shared_requested[j]) continue;
                        if (left_all[j].empty() || right_all[j].empty()) continue;
                        update_numerator_extrema_from_sum_(
                            acc[j],
                            std::move(left_all[j]),
                            std::move(right_all[j]),
                            n
                        );
                    }
                }
            }

            if (!do_split_variable) continue;

            const std::vector<int>* group_of_row_ptr =
                matched_group_of_row_by_variable_eval
                    ? &(*matched_group_of_row_by_variable_eval)[split_j]
                    : nullptr;
            const std::vector<int>* group_size_ptr =
                matched_group_size_by_variable_eval
                    ? &(*matched_group_size_by_variable_eval)[split_j]
                    : nullptr;
            const bool matched_effectively_uniform =
                matched_group_effectively_uniform_by_variable_eval &&
                (*matched_group_effectively_uniform_by_variable_eval)[split_j] != 0;

            L->ensure_hist_built();
            R->ensure_hist_built();

            const int max_left_budget =
                std::min(L->budget, budget - minR);
            const int max_right_budget =
                std::min(R->budget, budget - minL);

            std::size_t left_count = 0;
            for (const auto& e : L->hist) {
                if (e.obj > max_left_budget) break;
                ++left_count;
            }
            std::size_t right_count = 0;
            for (const auto& e : R->hist) {
                if (e.obj > max_right_budget) break;
                ++right_count;
            }

            ExactLocalImportanceNumeratorExtrema_ split_acc;

            if (left_count <= right_count) {
                for (const auto& e : L->hist) {
                    if (e.obj > max_left_budget) break;
                    const int bL = e.obj;
                    const int bR = std::min(R->budget, budget - bL);
                    if (bR < minR) continue;

                    auto left_ext =
                        collect_exact_local_importance_numerator_extrema_at_most_(
                            L,
                            bL,
                            original_left,
                            replacement_root_mask,
                            left_states[split_j],
                            split_variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            group_of_row_ptr,
                            group_size_ptr,
                            matched_effectively_uniform,
                            matched_scratch,
                            node_variable_masks
                        );
                    auto right_ext =
                        collect_exact_local_importance_numerator_extrema_at_most_(
                            R,
                            bR,
                            original_right,
                            replacement_root_mask,
                            right_states[split_j],
                            split_variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            group_of_row_ptr,
                            group_size_ptr,
                            matched_effectively_uniform,
                            matched_scratch,
                            node_variable_masks
                        );
                    if (left_ext.empty() || right_ext.empty()) continue;
                    update_numerator_extrema_from_sum_(
                        split_acc,
                        std::move(left_ext),
                        std::move(right_ext),
                        n
                    );
                }
            } else {
                for (const auto& e : R->hist) {
                    if (e.obj > max_right_budget) break;
                    const int bR = e.obj;
                    const int bL = std::min(L->budget, budget - bR);
                    if (bL < minL) continue;

                    auto left_ext =
                        collect_exact_local_importance_numerator_extrema_at_most_(
                            L,
                            bL,
                            original_left,
                            replacement_root_mask,
                            left_states[split_j],
                            split_variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            group_of_row_ptr,
                            group_size_ptr,
                            matched_effectively_uniform,
                            matched_scratch,
                            node_variable_masks
                        );
                    auto right_ext =
                        collect_exact_local_importance_numerator_extrema_at_most_(
                            R,
                            bR,
                            original_right,
                            replacement_root_mask,
                            right_states[split_j],
                            split_variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            group_of_row_ptr,
                            group_size_ptr,
                            matched_effectively_uniform,
                            matched_scratch,
                            node_variable_masks
                        );
                    if (left_ext.empty() || right_ext.empty()) continue;
                    update_numerator_extrema_from_sum_(
                        split_acc,
                        std::move(left_ext),
                        std::move(right_ext),
                        n
                    );
                }
            }

            if (!split_acc.empty()) {
                merge_numerator_extrema_move_(
                    acc[split_j],
                    std::move(split_acc),
                    n
                );
            }
        }

        return acc;
    }

    ExactLocalImportanceExtrema_
    collect_exact_local_importance_extrema_at_most_(
        const TreeTrieNode* node,
        int budget,
        const Packed& original_mask,
        const Packed& replacement_root_mask,
        const ExactReplacementState_& state,
        int variable,
        const std::vector<int>& internal_to_variable,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<int>* matched_group_of_row_eval,
        const std::vector<double>* matched_group_inv_size_eval,
        bool matched_effectively_uniform,
        ExactMatchedScratch_* matched_scratch
    ) const {
        ExactLocalImportanceExtrema_ acc;

        if (!node || budget < 0) return acc;

        constexpr int INF = std::numeric_limits<int>::max();

        if (
            node->min_objective == INF ||
            node->min_objective > budget
        ) {
            return acc;
        }

        const std::size_t n = static_cast<std::size_t>(ctx.n_eval);
        std::vector<double> scratch;

        auto ensure_acc = [&]() {
            if (!acc.lower.empty()) return;
            acc.lower.assign(n, std::numeric_limits<double>::infinity());
            acc.upper.assign(n, -std::numeric_limits<double>::infinity());
        };

        // OR alternative: leaf.
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            exact_local_importance_for_leaf_variable_into_(
                original_mask,
                state,
                leaf.prediction,
                ctx,
                Y_eval_bits,
                BBwrong_eval,
                matched_group_of_row_eval,
                matched_group_inv_size_eval,
                matched_effectively_uniform,
                matched_scratch,
                scratch
            );

            ensure_acc();
            fast_update_minmax_(
                acc.lower.data(),
                acc.upper.data(),
                scratch.data(),
                scratch.data(),
                n
            );
        }

        // OR alternatives: splits.
        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;
            if (minL == INF || minR == INF) continue;
            if (minL + minR > budget) continue;

            if (
                split.feature < 0 ||
                split.feature >=
                    static_cast<int>(internal_to_variable.size())
            ) {
                throw std::runtime_error(
                    "Exact local interval evaluation saw an invalid split feature."
                );
            }

            const int split_variable =
                internal_to_variable[
                    static_cast<std::size_t>(split.feature)
                ];

            if (split_variable < 0) {
                throw std::runtime_error(
                    "Split feature is not mapped to an original variable."
                );
            }

            const Packed& Xf =
                ctx.X_bits_eval[
                    static_cast<std::size_t>(split.feature)
                ];

            Packed original_left(static_cast<std::size_t>(ctx.n_words));
            Packed original_right(static_cast<std::size_t>(ctx.n_words));

            and_bits_eval(
                original_mask,
                Xf,
                original_left,
                ctx.n_words,
                ctx.tail_mask
            );
            andnot_bits_eval(
                original_mask,
                Xf,
                original_right,
                ctx.n_words,
                ctx.tail_mask
            );

            ExactReplacementState_ left_state;
            ExactReplacementState_ right_state;

            if (!state.replacement_feature_used) {
                if (split_variable == variable) {
                    left_state.replacement_feature_used = true;
                    right_state.replacement_feature_used = true;
                    left_state.target_rows = original_mask;
                    right_state.target_rows = original_mask;
                    left_state.replacement_values =
                        Packed(static_cast<std::size_t>(ctx.n_words));
                    right_state.replacement_values =
                        Packed(static_cast<std::size_t>(ctx.n_words));

                    and_bits_eval(
                        replacement_root_mask,
                        Xf,
                        left_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        replacement_root_mask,
                        Xf,
                        right_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                }
            } else {
                left_state.replacement_feature_used = true;
                right_state.replacement_feature_used = true;
                left_state.target_rows =
                    Packed(static_cast<std::size_t>(ctx.n_words));
                right_state.target_rows =
                    Packed(static_cast<std::size_t>(ctx.n_words));
                left_state.replacement_values =
                    Packed(static_cast<std::size_t>(ctx.n_words));
                right_state.replacement_values =
                    Packed(static_cast<std::size_t>(ctx.n_words));

                if (split_variable == variable) {
                    left_state.target_rows.w = state.target_rows.w;
                    right_state.target_rows.w = state.target_rows.w;

                    and_bits_eval(
                        state.replacement_values,
                        Xf,
                        left_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        state.replacement_values,
                        Xf,
                        right_state.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                } else {
                    and_bits_eval(
                        state.target_rows,
                        Xf,
                        left_state.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        state.target_rows,
                        Xf,
                        right_state.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    left_state.replacement_values.w =
                        state.replacement_values.w;
                    right_state.replacement_values.w =
                        state.replacement_values.w;
                }
            }

            // the major optimization
            // if this split is not on j, each evaluation row has nonzero local
            // contribution in at most one child. Therefore the sibling can be
            // fixed to its minimum-objective subtree, freeing all remaining
            // budget for the child that matters to that row.
            if (split_variable != variable) {
                const int left_budget =
                    std::min(L->budget, budget - minR);
                const int right_budget =
                    std::min(R->budget, budget - minL);

                auto left_ext =
                    collect_exact_local_importance_extrema_at_most_(
                        L,
                        left_budget,
                        original_left,
                        replacement_root_mask,
                        left_state,
                        variable,
                        internal_to_variable,
                        ctx,
                        Y_eval_bits,
                        BBwrong_eval,
                        matched_group_of_row_eval,
                        matched_group_inv_size_eval,
                        matched_effectively_uniform,
                        matched_scratch
                    );

                auto right_ext =
                    collect_exact_local_importance_extrema_at_most_(
                        R,
                        right_budget,
                        original_right,
                        replacement_root_mask,
                        right_state,
                        variable,
                        internal_to_variable,
                        ctx,
                        Y_eval_bits,
                        BBwrong_eval,
                        matched_group_of_row_eval,
                        matched_group_inv_size_eval,
                        matched_effectively_uniform,
                        matched_scratch
                    );

                if (left_ext.lower.empty() || right_ext.lower.empty()) continue;

                ensure_acc();
                fast_update_minmax_sum_(
                    acc.lower.data(),
                    acc.upper.data(),
                    left_ext.lower.data(),
                    left_ext.upper.data(),
                    right_ext.lower.data(),
                    right_ext.upper.data(),
                    n
                );
                continue;
            }

            // split is on j: the same target row can receive perturbed mass from
            // both children, so we do need a true budget convolution. 
            // we convolve only at these j-splits, and over at-most-budget extrema.
            L->ensure_hist_built();
            R->ensure_hist_built();

            const int max_left_budget =
                std::min(L->budget, budget - minR);
            const int max_right_budget =
                std::min(R->budget, budget - minL);

            std::size_t left_count = 0;
            for (const auto& e : L->hist) {
                if (e.obj > max_left_budget) break;
                ++left_count;
            }

            std::size_t right_count = 0;
            for (const auto& e : R->hist) {
                if (e.obj > max_right_budget) break;
                ++right_count;
            }

            ExactLocalImportanceExtrema_ split_acc;
            auto ensure_split_acc = [&]() {
                if (!split_acc.lower.empty()) return;
                split_acc.lower.assign(
                    n,
                    std::numeric_limits<double>::infinity()
                );
                split_acc.upper.assign(
                    n,
                    -std::numeric_limits<double>::infinity()
                );
            };

            if (left_count <= right_count) {
                for (const auto& e : L->hist) {
                    if (e.obj > max_left_budget) break;

                    const int left_budget = e.obj;
                    const int right_budget =
                        std::min(R->budget, budget - left_budget);
                    if (right_budget < minR) continue;

                    auto left_ext =
                        collect_exact_local_importance_extrema_at_most_(
                            L,
                            left_budget,
                            original_left,
                            replacement_root_mask,
                            left_state,
                            variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            matched_group_of_row_eval,
                            matched_group_inv_size_eval,
                            matched_effectively_uniform,
                            matched_scratch
                        );

                    auto right_ext =
                        collect_exact_local_importance_extrema_at_most_(
                            R,
                            right_budget,
                            original_right,
                            replacement_root_mask,
                            right_state,
                            variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            matched_group_of_row_eval,
                            matched_group_inv_size_eval,
                            matched_effectively_uniform,
                            matched_scratch
                        );

                    
                    if (
                        left_ext.lower.empty() ||
                        right_ext.lower.empty()
                    ) {
                        continue;
                    }

                    ensure_split_acc();
                    fast_update_minmax_sum_(
                        split_acc.lower.data(),
                        split_acc.upper.data(),
                        left_ext.lower.data(),
                        left_ext.upper.data(),
                        right_ext.lower.data(),
                        right_ext.upper.data(),
                        n
                    );
                }
            } else {

                for (const auto& e : R->hist) {
                    if (e.obj > max_right_budget) break;

                    const int right_budget = e.obj;
                    const int left_budget =
                        std::min(L->budget, budget - right_budget);
                    if (left_budget < minL) continue;

                    auto left_ext =
                        collect_exact_local_importance_extrema_at_most_(
                            L,
                            left_budget,
                            original_left,
                            replacement_root_mask,
                            left_state,
                            variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            matched_group_of_row_eval,
                            matched_group_inv_size_eval,
                            matched_effectively_uniform,
                            matched_scratch
                        );

                    auto right_ext =
                        collect_exact_local_importance_extrema_at_most_(
                            R,
                            right_budget,
                            original_right,
                            replacement_root_mask,
                            right_state,
                            variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            matched_group_of_row_eval,
                            matched_group_inv_size_eval,
                            matched_effectively_uniform,
                            matched_scratch
                        );

                    if (
                        left_ext.lower.empty() ||
                        right_ext.lower.empty()
                    ) {
                        continue;
                    }

                    ensure_split_acc();
                    fast_update_minmax_sum_(
                        split_acc.lower.data(),
                        split_acc.upper.data(),
                        left_ext.lower.data(),
                        left_ext.upper.data(),
                        right_ext.lower.data(),
                        right_ext.upper.data(),
                        n
                    );
                }
            }

            if (!split_acc.lower.empty()) {
                ensure_acc();
                fast_update_minmax_(
                    acc.lower.data(),
                    acc.upper.data(),
                    split_acc.lower.data(),
                    split_acc.upper.data(),
                    n
                );
            }
        }

        return acc;
    }


    ExactAllLocalImportanceExtrema_
    collect_exact_all_local_importance_extrema_at_most_(
        const TreeTrieNode* node,
        int budget,
        const Packed& original_mask,
        const Packed& replacement_root_mask,
        const std::vector<ExactReplacementState_>& states,
        const std::vector<int>& internal_to_variable,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<std::vector<int>>*
            matched_group_of_row_by_variable_eval,
        const std::vector<std::vector<double>>*
            matched_group_inv_size_by_variable_eval,
        const std::vector<uint8_t>*
            matched_group_effectively_uniform_by_variable_eval
    ) const {
        ExactAllLocalImportanceExtrema_ acc;

        if (!node || budget < 0) return acc;

        constexpr int INF = std::numeric_limits<int>::max();
        if (
            node->min_objective == INF ||
            node->min_objective > budget
        ) {
            return acc;
        }

        const int number_of_variables =
            static_cast<int>(states.size());
        const std::size_t n = static_cast<std::size_t>(ctx.n_eval);
        const std::size_t total =
            static_cast<std::size_t>(number_of_variables) * n;

        acc.number_of_variables = number_of_variables;
        acc.n_eval = ctx.n_eval;
        acc.lower.assign(total, std::numeric_limits<double>::infinity());
        acc.upper.assign(total, -std::numeric_limits<double>::infinity());

        std::vector<double> scratch;
        scratch.reserve(n);
        ExactMatchedScratch_ matched_scratch;

        // leaf alternatives. For variables not yet used on the path, every
        // feasible leaf has exactly zero local importance, so include zero once.
        bool has_feasible_leaf = false;
        for (const auto& leaf : node->leaves) {
            if (leaf.loss <= budget) {
                has_feasible_leaf = true;
                break;
            }
        }

        if (has_feasible_leaf) {
            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {
                if (!states[static_cast<std::size_t>(variable)]
                         .replacement_feature_used) {
                    fast_include_zero_(
                        acc.lower_ptr(variable),
                        acc.upper_ptr(variable),
                        n
                    );
                }
            }
        }

        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {
                const auto& state =
                    states[static_cast<std::size_t>(variable)];
                if (!state.replacement_feature_used) continue;

                const std::vector<int>* group_of_row_ptr = nullptr;
                const std::vector<double>* inv_group_size_ptr = nullptr;
                bool matched_effectively_uniform = false;
                ExactMatchedScratch_* matched_scratch_ptr = nullptr;

                if (matched_group_of_row_by_variable_eval != nullptr) {
                    group_of_row_ptr =
                        &(*matched_group_of_row_by_variable_eval)[
                            static_cast<std::size_t>(variable)
                        ];
                    inv_group_size_ptr =
                        &(*matched_group_inv_size_by_variable_eval)[
                            static_cast<std::size_t>(variable)
                        ];
                    matched_effectively_uniform =
                        (*matched_group_effectively_uniform_by_variable_eval)[
                            static_cast<std::size_t>(variable)
                        ] != 0;
                    matched_scratch_ptr = &matched_scratch;
                }

                exact_local_importance_for_leaf_variable_into_(
                    original_mask,
                    state,
                    leaf.prediction,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    group_of_row_ptr,
                    inv_group_size_ptr,
                    matched_effectively_uniform,
                    matched_scratch_ptr,
                    scratch
                );

                fast_update_minmax_(
                    acc.lower_ptr(variable),
                    acc.upper_ptr(variable),
                    scratch.data(),
                    scratch.data(),
                    n
                );
            }
        }

        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;
            if (minL == INF || minR == INF) continue;
            if (minL + minR > budget) continue;

            if (
                split.feature < 0 ||
                split.feature >=
                    static_cast<int>(internal_to_variable.size())
            ) {
                throw std::runtime_error(
                    "Exact all-variable local interval evaluation saw an invalid split feature."
                );
            }

            const int split_variable =
                internal_to_variable[
                    static_cast<std::size_t>(split.feature)
                ];
            if (
                split_variable < 0 ||
                split_variable >= number_of_variables
            ) {
                throw std::runtime_error(
                    "Split feature is not mapped to an original variable."
                );
            }

            const Packed& Xf =
                ctx.X_bits_eval[
                    static_cast<std::size_t>(split.feature)
                ];

            Packed original_left(static_cast<std::size_t>(ctx.n_words));
            Packed original_right(static_cast<std::size_t>(ctx.n_words));
            and_bits_eval(
                original_mask,
                Xf,
                original_left,
                ctx.n_words,
                ctx.tail_mask
            );
            andnot_bits_eval(
                original_mask,
                Xf,
                original_right,
                ctx.n_words,
                ctx.tail_mask
            );

            std::vector<ExactReplacementState_> left_states(
                static_cast<std::size_t>(number_of_variables)
            );
            std::vector<ExactReplacementState_> right_states(
                static_cast<std::size_t>(number_of_variables)
            );

            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {
                const auto& cur =
                    states[static_cast<std::size_t>(variable)];
                auto& ls =
                    left_states[static_cast<std::size_t>(variable)];
                auto& rs =
                    right_states[static_cast<std::size_t>(variable)];

                if (!cur.replacement_feature_used) {
                    if (variable != split_variable) continue;

                    ls.replacement_feature_used = true;
                    rs.replacement_feature_used = true;
                    ls.target_rows = original_mask;
                    rs.target_rows = original_mask;
                    ls.replacement_values =
                        Packed(static_cast<std::size_t>(ctx.n_words));
                    rs.replacement_values =
                        Packed(static_cast<std::size_t>(ctx.n_words));
                    and_bits_eval(
                        replacement_root_mask,
                        Xf,
                        ls.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        replacement_root_mask,
                        Xf,
                        rs.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    continue;
                }

                ls.replacement_feature_used = true;
                rs.replacement_feature_used = true;
                ls.target_rows = Packed(static_cast<std::size_t>(ctx.n_words));
                rs.target_rows = Packed(static_cast<std::size_t>(ctx.n_words));
                ls.replacement_values = Packed(static_cast<std::size_t>(ctx.n_words));
                rs.replacement_values = Packed(static_cast<std::size_t>(ctx.n_words));

                if (variable == split_variable) {
                    ls.target_rows.w = cur.target_rows.w;
                    rs.target_rows.w = cur.target_rows.w;
                    and_bits_eval(
                        cur.replacement_values,
                        Xf,
                        ls.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        cur.replacement_values,
                        Xf,
                        rs.replacement_values,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                } else {
                    and_bits_eval(
                        cur.target_rows,
                        Xf,
                        ls.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    andnot_bits_eval(
                        cur.target_rows,
                        Xf,
                        rs.target_rows,
                        ctx.n_words,
                        ctx.tail_mask
                    );
                    ls.replacement_values.w = cur.replacement_values.w;
                    rs.replacement_values.w = cur.replacement_values.w;
                }
            }

            const int left_budget =
                std::min(L->budget, budget - minR);
            const int right_budget =
                std::min(R->budget, budget - minL);

            // one batched traversal for every variable at the maximum budget each
            // side can receive when the sibling is fixed to its optimal objective.
            auto left_all =
                collect_exact_all_local_importance_extrema_at_most_(
                    L,
                    left_budget,
                    original_left,
                    replacement_root_mask,
                    left_states,
                    internal_to_variable,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    matched_group_of_row_by_variable_eval,
                    matched_group_inv_size_by_variable_eval,
                    matched_group_effectively_uniform_by_variable_eval
                );

            auto right_all =
                collect_exact_all_local_importance_extrema_at_most_(
                    R,
                    right_budget,
                    original_right,
                    replacement_root_mask,
                    right_states,
                    internal_to_variable,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval,
                    matched_group_of_row_by_variable_eval,
                    matched_group_inv_size_by_variable_eval,
                    matched_group_effectively_uniform_by_variable_eval
                );

            if (left_all.empty() || right_all.empty()) continue;

            // every feature except the split variable gets the optimal-sibling shortcut
            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {
                if (variable == split_variable) continue;

                fast_update_minmax_sum_(
                    acc.lower_ptr(variable),
                    acc.upper_ptr(variable),
                    left_all.lower_ptr(variable),
                    left_all.upper_ptr(variable),
                    right_all.lower_ptr(variable),
                    right_all.upper_ptr(variable),
                    n
                );
            }

            // only the split variable needs a true convolution (major optimzation)
            const auto& split_left_state =
                left_states[static_cast<std::size_t>(split_variable)];
            const auto& split_right_state =
                right_states[static_cast<std::size_t>(split_variable)];

            const std::vector<int>* group_of_row_ptr = nullptr;
            const std::vector<double>* inv_group_size_ptr = nullptr;
            bool matched_effectively_uniform = false;

            if (matched_group_of_row_by_variable_eval != nullptr) {
                group_of_row_ptr =
                    &(*matched_group_of_row_by_variable_eval)[
                        static_cast<std::size_t>(split_variable)
                    ];
                inv_group_size_ptr =
                    &(*matched_group_inv_size_by_variable_eval)[
                        static_cast<std::size_t>(split_variable)
                    ];
                matched_effectively_uniform =
                    (*matched_group_effectively_uniform_by_variable_eval)[
                        static_cast<std::size_t>(split_variable)
                    ] != 0;
            }

            L->ensure_hist_built();
            R->ensure_hist_built();

            const int max_left_budget = left_budget;
            const int max_right_budget = right_budget;

            std::size_t left_count = 0;
            for (const auto& e : L->hist) {
                if (e.obj > max_left_budget) break;
                ++left_count;
            }
            std::size_t right_count = 0;
            for (const auto& e : R->hist) {
                if (e.obj > max_right_budget) break;
                ++right_count;
            }

            ExactLocalImportanceExtrema_ split_ext;
            split_ext.lower.assign(
                n,
                std::numeric_limits<double>::infinity()
            );
            split_ext.upper.assign(
                n,
                -std::numeric_limits<double>::infinity()
            );
            bool saw_split_pair = false;
            ExactMatchedScratch_ split_matched_scratch;
            ExactMatchedScratch_* split_matched_scratch_ptr =
                matched_group_of_row_by_variable_eval != nullptr
                    ? &split_matched_scratch
                    : nullptr;

            if (left_count <= right_count) {
                int last_right_budget = -1;
                ExactLocalImportanceExtrema_ last_right_ext;

                for (const auto& e : L->hist) {
                    if (e.obj > max_left_budget) break;
                    const int lb = e.obj;
                    const int rb = std::min(R->budget, budget - lb);
                    if (rb < minR) continue;

                    auto lext =
                        collect_exact_local_importance_extrema_at_most_(
                            L,
                            lb,
                            original_left,
                            replacement_root_mask,
                            split_left_state,
                            split_variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            group_of_row_ptr,
                            inv_group_size_ptr,
                            matched_effectively_uniform,
                            split_matched_scratch_ptr
                        );

                    ExactLocalImportanceExtrema_ rext;
                    if (rb == last_right_budget) {
                        rext = last_right_ext;
                    } else {
                        rext =
                            collect_exact_local_importance_extrema_at_most_(
                                R,
                                rb,
                                original_right,
                                replacement_root_mask,
                                split_right_state,
                                split_variable,
                                internal_to_variable,
                                ctx,
                                Y_eval_bits,
                                BBwrong_eval,
                                group_of_row_ptr,
                                inv_group_size_ptr,
                                matched_effectively_uniform,
                                split_matched_scratch_ptr
                            );
                        last_right_budget = rb;
                        last_right_ext = rext;
                    }

                    if (lext.lower.empty() || rext.lower.empty()) continue;
                    saw_split_pair = true;
                    fast_update_minmax_sum_(
                        split_ext.lower.data(),
                        split_ext.upper.data(),
                        lext.lower.data(),
                        lext.upper.data(),
                        rext.lower.data(),
                        rext.upper.data(),
                        n
                    );
                }
            } else {
                int last_left_budget = -1;
                ExactLocalImportanceExtrema_ last_left_ext;

                for (const auto& e : R->hist) {
                    if (e.obj > max_right_budget) break;
                    const int rb = e.obj;
                    const int lb = std::min(L->budget, budget - rb);
                    if (lb < minL) continue;

                    ExactLocalImportanceExtrema_ lext;
                    if (lb == last_left_budget) {
                        lext = last_left_ext;
                    } else {
                        lext =
                            collect_exact_local_importance_extrema_at_most_(
                                L,
                                lb,
                                original_left,
                                replacement_root_mask,
                                split_left_state,
                                split_variable,
                                internal_to_variable,
                                ctx,
                                Y_eval_bits,
                                BBwrong_eval,
                                group_of_row_ptr,
                                inv_group_size_ptr,
                                matched_effectively_uniform,
                                split_matched_scratch_ptr
                            );
                        last_left_budget = lb;
                        last_left_ext = lext;
                    }

                    auto rext =
                        collect_exact_local_importance_extrema_at_most_(
                            R,
                            rb,
                            original_right,
                            replacement_root_mask,
                            split_right_state,
                            split_variable,
                            internal_to_variable,
                            ctx,
                            Y_eval_bits,
                            BBwrong_eval,
                            group_of_row_ptr,
                            inv_group_size_ptr,
                            matched_effectively_uniform,
                            split_matched_scratch_ptr
                        );

                    if (lext.lower.empty() || rext.lower.empty()) continue;
                    saw_split_pair = true;
                    fast_update_minmax_sum_(
                        split_ext.lower.data(),
                        split_ext.upper.data(),
                        lext.lower.data(),
                        lext.upper.data(),
                        rext.lower.data(),
                        rext.upper.data(),
                        n
                    );
                }
            }

            if (saw_split_pair) {
                fast_update_minmax_(
                    acc.lower_ptr(split_variable),
                    acc.upper_ptr(split_variable),
                    split_ext.lower.data(),
                    split_ext.upper.data(),
                    n
                );
            }
        }

        return acc;
    }


    struct ExactReplacementIntervalEvalSetup_ {
        EvalCtx ctx;
        int budget = 0;
        std::vector<std::vector<int>> variable_columns;
        std::vector<int> internal_to_variable;
        bool use_matched_groups = false;
        std::vector<std::vector<double>> matched_group_inv_sizes;
        std::vector<uint8_t> matched_group_effectively_uniform;
        Packed root_mask;
        std::vector<Packed> y_bits;
        Packed bb_wrong;
        bool has_bb_wrong = false;
    };

    ExactReplacementIntervalEvalSetup_
    prepare_exact_replacement_interval_eval_(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& y_eval,
        int budget_override,
        const std::vector<std::vector<int>>& variable_columns_in,
        const std::vector<int>& bb_pred_eval,
        const std::vector<std::vector<int>>&
            matched_group_of_row_by_variable_eval,
        const std::vector<std::vector<int>>&
            matched_group_size_by_variable_eval
    ) const {
        if (!result) {
            throw std::runtime_error(
                "No Rashomon trie has been constructed. Call fit() first."
            );
        }

        ExactReplacementIntervalEvalSetup_ out;

        out.ctx =
            build_eval_ctx_(
                X_row_major,
                this->n_features
            );

        if (static_cast<int>(y_eval.size()) != out.ctx.n_eval) {
            throw std::runtime_error(
                "Eval y has different number of rows than Eval X."
            );
        }

        out.budget =
            (budget_override >= 0)
                ? budget_override
                : result->budget;

        if (!variable_columns_in.empty()) {
            out.variable_columns = variable_columns_in;
        } else {
            const int first_cont = first_continuous_feature_();

            for (int f = 0; f < first_cont; ++f) {
                out.variable_columns.push_back({f});
            }

            for (int g = 0;
                 g < static_cast<int>(continuous_starts.size());
                 ++g) {

                const int start =
                    continuous_starts[static_cast<std::size_t>(g)];
                const int end = continuous_group_end_(g);

                std::vector<int> cols;
                cols.reserve(static_cast<std::size_t>(end - start));

                for (int f = start; f < end; ++f) {
                    cols.push_back(f);
                }

                out.variable_columns.push_back(std::move(cols));
            }
        }

        const int number_of_variables =
            static_cast<int>(out.variable_columns.size());

        const bool has_group_of_row =
            !matched_group_of_row_by_variable_eval.empty();
        const bool has_group_sizes =
            !matched_group_size_by_variable_eval.empty();

        if (has_group_of_row != has_group_sizes) {
            throw std::runtime_error(
                "Matched-group evaluation requires both group-of-row and "
                "group-size arrays."
            );
        }

        out.use_matched_groups = has_group_of_row;

        if (out.use_matched_groups) {
            if (
                static_cast<int>(
                    matched_group_of_row_by_variable_eval.size()
                ) != number_of_variables ||
                static_cast<int>(
                    matched_group_size_by_variable_eval.size()
                ) != number_of_variables
            ) {
                throw std::runtime_error(
                    "Matched-group arrays must contain exactly one entry "
                    "per original variable."
                );
            }

            out.matched_group_inv_sizes.resize(
                static_cast<std::size_t>(number_of_variables)
            );
            out.matched_group_effectively_uniform.assign(
                static_cast<std::size_t>(number_of_variables),
                0
            );

            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {

                const auto& group_of_row =
                    matched_group_of_row_by_variable_eval[
                        static_cast<std::size_t>(variable)
                    ];
                const auto& group_sizes =
                    matched_group_size_by_variable_eval[
                        static_cast<std::size_t>(variable)
                    ];

                if (static_cast<int>(group_of_row.size()) != out.ctx.n_eval) {
                    throw std::runtime_error(
                        "Matched-group row map has the wrong number of "
                        "evaluation rows."
                    );
                }

                if (group_sizes.empty()) {
                    throw std::runtime_error(
                        "Matched-group size array is empty."
                    );
                }

                std::vector<int> observed_group_sizes(
                    group_sizes.size(),
                    0
                );

                for (int row = 0; row < out.ctx.n_eval; ++row) {
                    const int group =
                        group_of_row[static_cast<std::size_t>(row)];

                    if (
                        group < 0 ||
                        group >= static_cast<int>(group_sizes.size())
                    ) {
                        throw std::runtime_error(
                            "Matched-group row map contains an out-of-range "
                            "group ID."
                        );
                    }

                    ++observed_group_sizes[
                        static_cast<std::size_t>(group)
                    ];
                }

                auto& inv =
                    out.matched_group_inv_sizes[
                        static_cast<std::size_t>(variable)
                    ];
                inv.assign(group_sizes.size(), 0.0);

                int number_of_nonempty_groups = 0;

                for (std::size_t group = 0;
                     group < group_sizes.size();
                     ++group) {

                    if (group_sizes[group] < 0) {
                        throw std::runtime_error(
                            "Matched-group size cannot be negative."
                        );
                    }

                    if (observed_group_sizes[group] != group_sizes[group]) {
                        throw std::runtime_error(
                            "Matched-group size array does not agree with the "
                            "row-to-group map."
                        );
                    }

                    if (group_sizes[group] > 0) {
                        inv[group] =
                            1.0 /
                            static_cast<double>(group_sizes[group]);
                        ++number_of_nonempty_groups;
                    }
                }

                if (number_of_nonempty_groups <= 0) {
                    throw std::runtime_error(
                        "Matched-group partition has no rows."
                    );
                }

                out.matched_group_effectively_uniform[
                    static_cast<std::size_t>(variable)
                ] =
                    number_of_nonempty_groups == 1 ? 1 : 0;
            }
        }

        out.internal_to_variable.assign(
            static_cast<std::size_t>(this->n_features),
            -1
        );

        for (int variable = 0;
             variable < number_of_variables;
             ++variable) {

            const auto& cols =
                out.variable_columns[static_cast<std::size_t>(variable)];

            if (cols.empty()) {
                throw std::runtime_error(
                    "variable_columns contains an empty variable."
                );
            }

            for (int f : cols) {
                if (f < 0 || f >= this->n_features) {
                    throw std::runtime_error(
                        "variable_columns contains an out-of-range internal "
                        "column."
                    );
                }

                if (
                    out.internal_to_variable[
                        static_cast<std::size_t>(f)
                    ] != -1
                ) {
                    throw std::runtime_error(
                        "An internal column appears in more than one variable."
                    );
                }

                out.internal_to_variable[
                    static_cast<std::size_t>(f)
                ] = variable;
            }
        }

        for (int f = 0; f < this->n_features; ++f) {
            if (
                out.internal_to_variable[
                    static_cast<std::size_t>(f)
                ] < 0
            ) {
                throw std::runtime_error(
                    "Every internal feature column must belong to exactly "
                    "one variable."
                );
            }
        }

        out.root_mask =
            eval_root_mask_(
                out.ctx.n_words,
                out.ctx.tail_mask
            );

        out.y_bits =
            build_eval_y_bits_(
                y_eval,
                num_classes,
                out.ctx.n_words,
                out.ctx.tail_mask
            );

        if (use_deferral) {
            if (bb_pred_eval.empty()) {
                throw std::runtime_error(
                    "Deferral was enabled during fit, so bb_pred_eval is required."
                );
            }

            out.bb_wrong =
                build_eval_bb_wrong_bits_(
                    y_eval,
                    bb_pred_eval,
                    num_classes,
                    out.ctx.n_words,
                    out.ctx.tail_mask
                );
            out.has_bb_wrong = true;
        }

        return out;
    }

    struct ExactImportanceFrontierPoint_ {
        int obj = 0;
        double value = 0.0;
    };

    using ExactImportanceFrontier_ =
        std::vector<ExactImportanceFrontierPoint_>;

    struct ExactSparseFeatureImportanceFrontiers_ {
        int variable = -1;
        ExactImportanceFrontier_ lower;
        ExactImportanceFrontier_ upper;
    };

    // sparse by variable. a variable is absent iff its importance is
    // identically zero for every feasible completion represented here.
    struct ExactGlobalImportanceFrontiers_ {
        int min_feasible_obj = std::numeric_limits<int>::max();
        std::vector<ExactSparseFeatureImportanceFrontiers_> features;
    };

    struct ExactImportanceBinaryConstraint_ {
        int feature = -1; // internal binary feature column
        int8_t value = -1; // 0 = false/right, 1 = true/left

        bool operator==(const ExactImportanceBinaryConstraint_& o) const {
            return feature == o.feature && value == o.value;
        }
    };

    struct ExactImportanceContinuousConstraint_ {
        int group = -1; // position in continuous_starts
        int lo = -1; // canonical active-threshold interval [lo, hi)
        int hi = -1;

        bool operator==(const ExactImportanceContinuousConstraint_& o) const {
            return group == o.group && lo == o.lo && hi == o.hi;
        }
    };

    // only constrained features/groups are stored. both vectors are kept sorted
    struct ExactImportanceSemanticPath_ {
        std::vector<ExactImportanceBinaryConstraint_> binary;
        std::vector<ExactImportanceContinuousConstraint_> continuous;
    };

    // only variables whose replacement routing has actually been activated
    // on the path are materialized. sorted by original-variable id.
    struct ExactSparseReplacementState_ {
        int variable = -1;
        ExactReplacementState_ state;
    };

    using ExactSparseReplacementStates_ =
        std::vector<ExactSparseReplacementState_>;

    struct ExactGlobalImportanceFrontierCacheEntry_ {
        // the entry contains all frontier breakpoints through this budget.
        // it may therefore answer any later query with a smaller budget.
        int solved_budget = -1;
        ExactGlobalImportanceFrontiers_ frontiers;
    };

    using ExactGlobalImportanceFrontierCache_ =
        std::unordered_map<
            std::string,
            std::shared_ptr<ExactGlobalImportanceFrontierCacheEntry_>
        >;

    struct ExactFeatureFrontierAccumulator_ {
        std::map<int, double> lower;
        std::map<int, double> upper;
    };

    template <typename T>
    static inline void append_exact_importance_key_bytes_(
        std::string& key,
        const T& value
    ) {
        const char* p = reinterpret_cast<const char*>(&value);
        key.append(p, sizeof(T));
    }

    std::string exact_importance_semantic_cache_key_(
        const ExactImportanceSemanticPath_& path,
        int remaining_depth
    ) const {
        std::string key;

        const uint32_t number_of_binary =
            static_cast<uint32_t>(path.binary.size());
        const uint32_t number_of_continuous =
            static_cast<uint32_t>(path.continuous.size());

        key.reserve(
            sizeof(int) +
            sizeof(uint32_t) * 2 +
            path.binary.size() * (sizeof(int) + sizeof(int8_t)) +
            path.continuous.size() * 3 * sizeof(int)
        );

        append_exact_importance_key_bytes_(key, remaining_depth);
        append_exact_importance_key_bytes_(key, number_of_binary);

        for (const auto& c : path.binary) {
            append_exact_importance_key_bytes_(key, c.feature);
            append_exact_importance_key_bytes_(key, c.value);
        }

        append_exact_importance_key_bytes_(key, number_of_continuous);

        for (const auto& c : path.continuous) {
            append_exact_importance_key_bytes_(key, c.group);
            append_exact_importance_key_bytes_(key, c.lo);
            append_exact_importance_key_bytes_(key, c.hi);
        }

        return key;
    }

    static inline void set_exact_importance_binary_constraint_(
        std::vector<ExactImportanceBinaryConstraint_>& constraints,
        int feature,
        int8_t value
    ) {
        auto it = std::lower_bound(
            constraints.begin(),
            constraints.end(),
            feature,
            [](const ExactImportanceBinaryConstraint_& c, int f) {
                return c.feature < f;
            }
        );

        if (it != constraints.end() && it->feature == feature) {
            if (it->value != value) {
                throw std::runtime_error(
                    "Cached global importance reached contradictory binary "
                    "path constraints."
                );
            }
            return;
        }

        constraints.insert(
            it,
            ExactImportanceBinaryConstraint_{feature, value}
        );
    }

    void set_exact_importance_continuous_constraint_(
        std::vector<ExactImportanceContinuousConstraint_>& constraints,
        int split_feature,
        bool take_left
    ) const {
        const int group =
            continuous_group_pos_for_threshold_(split_feature);

        if (group < 0) {
            throw std::runtime_error(
                "Cached global importance could not map a continuous split "
                "to its continuous group."
            );
        }

        const int default_lo =
            continuous_starts[static_cast<std::size_t>(group)];
        const int default_hi = continuous_group_end_(group);

        auto it = std::lower_bound(
            constraints.begin(),
            constraints.end(),
            group,
            [](const ExactImportanceContinuousConstraint_& c, int g) {
                return c.group < g;
            }
        );

        int lo = default_lo;
        int hi = default_hi;

        if (it != constraints.end() && it->group == group) {
            lo = it->lo;
            hi = it->hi;
        }

        // based on ContinuousPath:
        // left/true tightens hi to split_feature;
        // right/false tightens lo to split_feature + 1.
        if (take_left) {
            hi = std::min(hi, split_feature);
        } else {
            lo = std::max(lo, split_feature + 1);
        }

        if (it != constraints.end() && it->group == group) {
            it->lo = lo;
            it->hi = hi;
        } else {
            constraints.insert(
                it,
                ExactImportanceContinuousConstraint_{group, lo, hi}
            );
        }
    }

    void make_exact_importance_child_semantic_paths_(
        int split_feature,
        const ExactImportanceSemanticPath_& parent,
        ExactImportanceSemanticPath_& left,
        ExactImportanceSemanticPath_& right
    ) const {
        left = parent;
        right = parent;

        if (is_continuous_threshold_feature_(split_feature)) {
            set_exact_importance_continuous_constraint_(
                left.continuous,
                split_feature,
                true
            );
            set_exact_importance_continuous_constraint_(
                right.continuous,
                split_feature,
                false
            );
            return;
        }

        set_exact_importance_binary_constraint_(
            left.binary,
            split_feature,
            1
        );
        set_exact_importance_binary_constraint_(
            right.binary,
            split_feature,
            0
        );
    }

    static inline const ExactReplacementState_*
    find_exact_sparse_replacement_state_(
        const ExactSparseReplacementStates_& states,
        int variable
    ) {
        auto it = std::lower_bound(
            states.begin(),
            states.end(),
            variable,
            [](const ExactSparseReplacementState_& s, int v) {
                return s.variable < v;
            }
        );

        if (it == states.end() || it->variable != variable) {
            return nullptr;
        }
        return &it->state;
    }

    void make_exact_importance_child_sparse_states_(
        int split_variable,
        const Packed& Xf,
        const Packed& original_mask,
        const Packed& replacement_root_mask,
        const ExactSparseReplacementStates_& states,
        const EvalCtx& ctx,
        ExactSparseReplacementStates_& left_states,
        ExactSparseReplacementStates_& right_states
    ) const {
        left_states.clear();
        right_states.clear();
        left_states.reserve(states.size() + 1);
        right_states.reserve(states.size() + 1);

        bool saw_split_variable = false;

        for (const auto& sparse : states) {
            const int variable = sparse.variable;
            const auto& cur = sparse.state;

            if (!cur.replacement_feature_used) {
                throw std::runtime_error(
                    "Sparse replacement state contained an inactive entry."
                );
            }

            ExactReplacementState_ ls;
            ExactReplacementState_ rs;
            ls.replacement_feature_used = true;
            rs.replacement_feature_used = true;

            ls.target_rows = Packed(
                static_cast<std::size_t>(ctx.n_words)
            );
            rs.target_rows = Packed(
                static_cast<std::size_t>(ctx.n_words)
            );
            ls.replacement_values = Packed(
                static_cast<std::size_t>(ctx.n_words)
            );
            rs.replacement_values = Packed(
                static_cast<std::size_t>(ctx.n_words)
            );

            if (variable == split_variable) {
                saw_split_variable = true;

                ls.target_rows.w = cur.target_rows.w;
                rs.target_rows.w = cur.target_rows.w;

                and_bits_eval(
                    cur.replacement_values,
                    Xf,
                    ls.replacement_values,
                    ctx.n_words,
                    ctx.tail_mask
                );
                andnot_bits_eval(
                    cur.replacement_values,
                    Xf,
                    rs.replacement_values,
                    ctx.n_words,
                    ctx.tail_mask
                );
            } else {
                and_bits_eval(
                    cur.target_rows,
                    Xf,
                    ls.target_rows,
                    ctx.n_words,
                    ctx.tail_mask
                );
                andnot_bits_eval(
                    cur.target_rows,
                    Xf,
                    rs.target_rows,
                    ctx.n_words,
                    ctx.tail_mask
                );

                ls.replacement_values.w = cur.replacement_values.w;
                rs.replacement_values.w = cur.replacement_values.w;
            }

            left_states.push_back(
                ExactSparseReplacementState_{variable, std::move(ls)}
            );
            right_states.push_back(
                ExactSparseReplacementState_{variable, std::move(rs)}
            );
        }

        if (!saw_split_variable) {
            ExactReplacementState_ ls;
            ExactReplacementState_ rs;
            ls.replacement_feature_used = true;
            rs.replacement_feature_used = true;

            // first time this replacement variable appears on the path:
            // target rows are not split by their own value; donor values are.
            ls.target_rows = original_mask;
            rs.target_rows = original_mask;
            ls.replacement_values = Packed(
                static_cast<std::size_t>(ctx.n_words)
            );
            rs.replacement_values = Packed(
                static_cast<std::size_t>(ctx.n_words)
            );

            and_bits_eval(
                replacement_root_mask,
                Xf,
                ls.replacement_values,
                ctx.n_words,
                ctx.tail_mask
            );
            andnot_bits_eval(
                replacement_root_mask,
                Xf,
                rs.replacement_values,
                ctx.n_words,
                ctx.tail_mask
            );

            auto lit = std::lower_bound(
                left_states.begin(),
                left_states.end(),
                split_variable,
                [](const ExactSparseReplacementState_& s, int v) {
                    return s.variable < v;
                }
            );
            const std::size_t pos =
                static_cast<std::size_t>(
                    std::distance(left_states.begin(), lit)
                );

            left_states.insert(
                lit,
                ExactSparseReplacementState_{
                    split_variable,
                    std::move(ls)
                }
            );
            right_states.insert(
                right_states.begin() + static_cast<std::ptrdiff_t>(pos),
                ExactSparseReplacementState_{
                    split_variable,
                    std::move(rs)
                }
            );
        }
    }

    // maintain an at-most-budget lower frontier in-place.
    // frontier invariant: objective strictly increases and value strictly decreases.
    static inline void insert_exact_min_frontier_point_(
        std::map<int, double>& frontier,
        int obj,
        double value
    ) {
        auto it = frontier.lower_bound(obj);

        if (it != frontier.end() && it->first == obj) {
            if (it->second <= value) {
                return;
            }
            it->second = value;
        } else {
            if (it != frontier.begin()) {
                const auto prev = std::prev(it);
                if (prev->second <= value) {
                    return;
                }
            }
            it = frontier.emplace_hint(it, obj, value);
        }

        if (it != frontier.begin()) {
            const auto prev = std::prev(it);
            if (prev->second <= it->second) {
                frontier.erase(it);
                return;
            }
        }

        auto next = std::next(it);
        while (
            next != frontier.end() &&
            next->second >= it->second
        ) {
            next = frontier.erase(next);
        }
    }

    // maintain an at-most-budget upper frontier in-place.
    // frontier invariant: objective strictly increases and value strictly increases.
    static inline void insert_exact_max_frontier_point_(
        std::map<int, double>& frontier,
        int obj,
        double value
    ) {
        auto it = frontier.lower_bound(obj);

        if (it != frontier.end() && it->first == obj) {
            if (it->second >= value) {
                return;
            }
            it->second = value;
        } else {
            if (it != frontier.begin()) {
                const auto prev = std::prev(it);
                if (prev->second >= value) {
                    return;
                }
            }
            it = frontier.emplace_hint(it, obj, value);
        }

        if (it != frontier.begin()) {
            const auto prev = std::prev(it);
            if (prev->second >= it->second) {
                frontier.erase(it);
                return;
            }
        }

        auto next = std::next(it);
        while (
            next != frontier.end() &&
            next->second <= it->second
        ) {
            next = frontier.erase(next);
        }
    }

    static inline ExactImportanceFrontier_
    exact_frontier_map_to_vector_(
        const std::map<int, double>& frontier
    ) {
        ExactImportanceFrontier_ out;
        out.reserve(frontier.size());

        for (const auto& [obj, value] : frontier) {
            out.push_back(ExactImportanceFrontierPoint_{obj, value});
        }

        return out;
    }

    static inline double exact_frontier_value_at_budget_(
        const ExactImportanceFrontier_& frontier,
        int budget
    ) {
        const auto it = std::upper_bound(
            frontier.begin(),
            frontier.end(),
            budget,
            [](int b, const ExactImportanceFrontierPoint_& p) {
                return b < p.obj;
            }
        );

        if (it == frontier.begin()) {
            throw std::runtime_error(
                "No cached importance-frontier point is feasible at the "
                "requested budget."
            );
        }

        return std::prev(it)->value;
    }

    static inline const ExactSparseFeatureImportanceFrontiers_*
    find_exact_sparse_feature_frontiers_(
        const ExactGlobalImportanceFrontiers_& frontiers,
        int variable
    ) {
        auto it = std::lower_bound(
            frontiers.features.begin(),
            frontiers.features.end(),
            variable,
            [](const ExactSparseFeatureImportanceFrontiers_& f, int v) {
                return f.variable < v;
            }
        );

        if (
            it == frontiers.features.end() ||
            it->variable != variable
        ) {
            return nullptr;
        }
        return &(*it);
    }

    static inline std::vector<int>
    exact_sparse_feature_union_(
        const ExactGlobalImportanceFrontiers_& left,
        const ExactGlobalImportanceFrontiers_& right
    ) {
        std::vector<int> out;
        out.reserve(left.features.size() + right.features.size());

        std::size_t li = 0;
        std::size_t ri = 0;

        while (
            li < left.features.size() ||
            ri < right.features.size()
        ) {
            if (
                ri >= right.features.size() ||
                (li < left.features.size() &&
                 left.features[li].variable < right.features[ri].variable)
            ) {
                out.push_back(left.features[li].variable);
                ++li;
            } else if (
                li >= left.features.size() ||
                right.features[ri].variable < left.features[li].variable
            ) {
                out.push_back(right.features[ri].variable);
                ++ri;
            } else {
                out.push_back(left.features[li].variable);
                ++li;
                ++ri;
            }
        }

        return out;
    }

    static inline bool packed_equal_exact_(
        const Packed& a,
        const Packed& b
    ) {
        return a.w == b.w;
    }

    void require_exact_importance_eval_is_training_(
        const ExactReplacementIntervalEvalSetup_& setup,
        const std::vector<int>& y_eval
    ) const {
        if (setup.ctx.n_eval != n_samples) {
            throw std::runtime_error(
                "Cached global importance currently requires evaluation data "
                "to be exactly the training data (row count differs)."
            );
        }

        if (setup.ctx.X_bits_eval.size() != X_bits.size()) {
            throw std::runtime_error(
                "Cached global importance currently requires evaluation X "
                "to be exactly the training X."
            );
        }

        for (std::size_t f = 0; f < X_bits.size(); ++f) {
            if (!packed_equal_exact_(setup.ctx.X_bits_eval[f], X_bits[f])) {
                throw std::runtime_error(
                    "Cached global importance currently requires evaluation X "
                    "to be exactly the training X."
                );
            }
        }

        if (y_eval != y_train) {
            throw std::runtime_error(
                "Cached global importance currently requires evaluation y "
                "to be exactly the training y."
            );
        }

        if (
            use_deferral &&
            (!setup.has_bb_wrong ||
             !packed_equal_exact_(setup.bb_wrong, BBwrong))
        ) {
            throw std::runtime_error(
                "Cached global importance with deferral requires evaluation "
                "black-box predictions to match the training black-box "
                "predictions."
            );
        }
    }

    static inline void insert_zero_for_sparse_feature_accumulator_(
        ExactFeatureFrontierAccumulator_& acc,
        int obj
    ) {
        if (obj == std::numeric_limits<int>::max()) return;
        insert_exact_min_frontier_point_(acc.lower, obj, 0.0);
        insert_exact_max_frontier_point_(acc.upper, obj, 0.0);
    }

    std::shared_ptr<const ExactGlobalImportanceFrontierCacheEntry_>
    collect_exact_global_importance_frontiers_cached_(
        const TreeTrieNode* node,
        int remaining_depth,
        int delta,
        const Packed& original_mask,
        const Packed& replacement_root_mask,
        const ExactSparseReplacementStates_& states,
        const ExactImportanceSemanticPath_& semantic_path,
        const std::vector<int>& internal_to_variable,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval,
        const std::vector<std::vector<int>>*
            matched_group_of_row_by_variable_eval,
        const std::vector<std::vector<double>>*
            matched_group_inv_size_by_variable_eval,
        const std::vector<uint8_t>*
            matched_group_effectively_uniform_by_variable_eval,
        ExactMatchedScratch_* matched_scratch,
        ExactGlobalImportanceFrontierCache_& cache
    ) const {
        if (!node || remaining_depth < 0) {
            return nullptr;
        }

        constexpr int INF = std::numeric_limits<int>::max();

        // The canonical node stores the largest budget ever solved there.
        // Reducing the root budget by delta shifts that largest reachable
        // node budget down by the same delta.
        const int budget = node->budget - delta;

        if (
            budget < 0 ||
            node->min_objective == INF ||
            node->min_objective > budget
        ) {
            return nullptr;
        }

        const std::string cache_key =
            exact_importance_semantic_cache_key_(
                semantic_path,
                remaining_depth
            );

        if (auto it = cache.find(cache_key); it != cache.end()) {
            if (it->second && it->second->solved_budget >= budget) {
                return it->second;
            }
        }

        // Sparse by variable: a feature accumulator is created only when
        // that variable is already active on the path or is discovered in
        // a feasible descendant split.
        std::map<int, ExactFeatureFrontierAccumulator_> feature_acc;

        // Minimum objective among OR alternatives processed so far.  When a
        // previously unseen variable is discovered later, all earlier
        // alternatives had zero importance for it, so this single number is
        // sufficient to seed its zero point exactly.
        int prior_min_feasible_obj = INF;
        bool saw_solution = false;

        // OR alternatives: leaves.
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            const int original_mistakes =
                exact_wrong_count_for_leaf_(
                    original_mask,
                    leaf.prediction,
                    ctx,
                    Y_eval_bits,
                    BBwrong_eval
                );

            // existing sparse parent variables that are not active on this
            // path get exactly zero importance for this leaf alternative.
            for (auto& [variable, acc] : feature_acc) {
                if (
                    find_exact_sparse_replacement_state_(
                        states,
                        variable
                    ) == nullptr
                ) {
                    insert_zero_for_sparse_feature_accumulator_(
                        acc,
                        leaf.loss
                    );
                }
            }

            // only variables already activated by an ancestor split need a leaf importance calculation
            for (const auto& sparse : states) {
                const int variable = sparse.variable;
                const auto& state = sparse.state;

                auto [it, inserted] =
                    feature_acc.try_emplace(variable);

                // if this variable is first materialized only now, every
                // previously processed OR alternative was zero for it.
                if (inserted && prior_min_feasible_obj != INF) {
                    insert_zero_for_sparse_feature_accumulator_(
                        it->second,
                        prior_min_feasible_obj
                    );
                }

                const double replacement_mistakes =
                    exact_replacement_expected_mistakes_for_leaf_variable_(
                        state,
                        variable,
                        leaf.prediction,
                        original_mistakes,
                        ctx,
                        Y_eval_bits,
                        BBwrong_eval,
                        matched_group_of_row_by_variable_eval,
                        matched_group_inv_size_by_variable_eval,
                        matched_group_effectively_uniform_by_variable_eval,
                        matched_scratch
                    );

                const double importance =
                    replacement_mistakes -
                    static_cast<double>(original_mistakes);

                insert_exact_min_frontier_point_(
                    it->second.lower,
                    leaf.loss,
                    importance
                );
                insert_exact_max_frontier_point_(
                    it->second.upper,
                    leaf.loss,
                    importance
                );
            }

            prior_min_feasible_obj =
                std::min(prior_min_feasible_obj, leaf.loss);
            saw_solution = true;
        }

        // OR alternatives: split choices.
        if (remaining_depth > 0) {
            for (const auto& split : node->splits) {
                const TreeTrieNode* L = split.left.get();
                const TreeTrieNode* R = split.right.get();

                if (!L || !R) continue;

                const int minL = L->min_objective;
                const int minR = R->min_objective;

                if (minL == INF || minR == INF) continue;
                if (minL + minR > budget) continue;

                if (
                    split.feature < 0 ||
                    split.feature >=
                        static_cast<int>(internal_to_variable.size())
                ) {
                    throw std::runtime_error(
                        "Cached global importance saw an invalid split feature."
                    );
                }

                const int split_variable =
                    internal_to_variable[
                        static_cast<std::size_t>(split.feature)
                    ];

                if (split_variable < 0) {
                    throw std::runtime_error(
                        "Cached global importance split feature is not mapped "
                        "to an original variable."
                    );
                }

                const Packed& Xf =
                    ctx.X_bits_eval[
                        static_cast<std::size_t>(split.feature)
                    ];

                Packed original_left(
                    static_cast<std::size_t>(ctx.n_words)
                );
                Packed original_right(
                    static_cast<std::size_t>(ctx.n_words)
                );

                and_bits_eval(
                    original_mask,
                    Xf,
                    original_left,
                    ctx.n_words,
                    ctx.tail_mask
                );
                andnot_bits_eval(
                    original_mask,
                    Xf,
                    original_right,
                    ctx.n_words,
                    ctx.tail_mask
                );

                ExactSparseReplacementStates_ left_states;
                ExactSparseReplacementStates_ right_states;

                make_exact_importance_child_sparse_states_(
                    split_variable,
                    Xf,
                    original_mask,
                    replacement_root_mask,
                    states,
                    ctx,
                    left_states,
                    right_states
                );

                ExactImportanceSemanticPath_ left_path;
                ExactImportanceSemanticPath_ right_path;

                make_exact_importance_child_semantic_paths_(
                    split.feature,
                    semantic_path,
                    left_path,
                    right_path
                );

                // deliberately do not subtract the sibling minimum objective before recursion. 
                // use canonical root node budget - delta for all nodes, solving to exactly what you would need for any way you reach it
                auto left_entry =
                    collect_exact_global_importance_frontiers_cached_(
                        L,
                        remaining_depth - 1,
                        delta,
                        original_left,
                        replacement_root_mask,
                        left_states,
                        left_path,
                        internal_to_variable,
                        ctx,
                        Y_eval_bits,
                        BBwrong_eval,
                        matched_group_of_row_by_variable_eval,
                        matched_group_inv_size_by_variable_eval,
                        matched_group_effectively_uniform_by_variable_eval,
                        matched_scratch,
                        cache
                    );

                auto right_entry =
                    collect_exact_global_importance_frontiers_cached_(
                        R,
                        remaining_depth - 1,
                        delta,
                        original_right,
                        replacement_root_mask,
                        right_states,
                        right_path,
                        internal_to_variable,
                        ctx,
                        Y_eval_bits,
                        BBwrong_eval,
                        matched_group_of_row_by_variable_eval,
                        matched_group_inv_size_by_variable_eval,
                        matched_group_effectively_uniform_by_variable_eval,
                        matched_scratch,
                        cache
                    );

                if (!left_entry || !right_entry) continue;

                const auto& LF = left_entry->frontiers;
                const auto& RF = right_entry->frontiers;

                const int left_budget = L->budget - delta;
                const int right_budget = R->budget - delta;

                if (
                    LF.min_feasible_obj == INF ||
                    RF.min_feasible_obj == INF ||
                    LF.min_feasible_obj > left_budget ||
                    RF.min_feasible_obj > right_budget
                ) {
                    continue;
                }

                const int split_min_obj =
                    LF.min_feasible_obj + RF.min_feasible_obj;

                if (split_min_obj > budget) continue;

                const std::vector<int> split_variables =
                    exact_sparse_feature_union_(LF, RF);

                // any parent feature already known but absent from this split
                // is identically zero for every completion under this split.
                for (auto& [variable, acc] : feature_acc) {
                    if (
                        !std::binary_search(
                            split_variables.begin(),
                            split_variables.end(),
                            variable
                        )
                    ) {
                        insert_zero_for_sparse_feature_accumulator_(
                            acc,
                            split_min_obj
                        );
                    }
                }

                for (int variable : split_variables) {
                    const auto* left_feature =
                        find_exact_sparse_feature_frontiers_(
                            LF,
                            variable
                        );
                    const auto* right_feature =
                        find_exact_sparse_feature_frontiers_(
                            RF,
                            variable
                        );

                    auto [it, inserted] =
                        feature_acc.try_emplace(variable);

                    // this feature was absent from every earlier OR
                    // alternative, hence those alternatives contribute zero.
                    if (inserted && prior_min_feasible_obj != INF) {
                        insert_zero_for_sparse_feature_accumulator_(
                            it->second,
                            prior_min_feasible_obj
                        );
                    }

                    auto& parent_feature = it->second;

                    if (left_feature && right_feature) {
                        // AND convolution for the minimum frontier
                        for (const auto& lp : left_feature->lower) {
                            if (lp.obj > left_budget) break;
                            if (lp.obj > budget) break;

                            const int rem_parent = budget - lp.obj;
                            const int rem =
                                std::min(rem_parent, right_budget);

                            for (const auto& rp : right_feature->lower) {
                                if (rp.obj > rem) break;

                                insert_exact_min_frontier_point_(
                                    parent_feature.lower,
                                    lp.obj + rp.obj,
                                    lp.value + rp.value
                                );
                            }
                        }

                        // AND convolution for the maximum frontier
                        for (const auto& lp : left_feature->upper) {
                            if (lp.obj > left_budget) break;
                            if (lp.obj > budget) break;

                            const int rem_parent = budget - lp.obj;
                            const int rem =
                                std::min(rem_parent, right_budget);

                            for (const auto& rp : right_feature->upper) {
                                if (rp.obj > rem) break;

                                insert_exact_max_frontier_point_(
                                    parent_feature.upper,
                                    lp.obj + rp.obj,
                                    lp.value + rp.value
                                );
                            }
                        }
                    } else if (left_feature) {
                        // right side is identically zero for this feature.
                        // only its cheapest feasible completion can matter.
                        const int zero_obj = RF.min_feasible_obj;

                        for (const auto& lp : left_feature->lower) {
                            if (lp.obj > left_budget) break;
                            const int total_obj = lp.obj + zero_obj;
                            if (total_obj > budget) break;

                            insert_exact_min_frontier_point_(
                                parent_feature.lower,
                                total_obj,
                                lp.value
                            );
                        }

                        for (const auto& lp : left_feature->upper) {
                            if (lp.obj > left_budget) break;
                            const int total_obj = lp.obj + zero_obj;
                            if (total_obj > budget) break;

                            insert_exact_max_frontier_point_(
                                parent_feature.upper,
                                total_obj,
                                lp.value
                            );
                        }
                    } else if (right_feature) {
                        // left side is identically zero for this feature
                        const int zero_obj = LF.min_feasible_obj;

                        for (const auto& rp : right_feature->lower) {
                            if (rp.obj > right_budget) break;
                            const int total_obj = zero_obj + rp.obj;
                            if (total_obj > budget) break;

                            insert_exact_min_frontier_point_(
                                parent_feature.lower,
                                total_obj,
                                rp.value
                            );
                        }

                        for (const auto& rp : right_feature->upper) {
                            if (rp.obj > right_budget) break;
                            const int total_obj = zero_obj + rp.obj;
                            if (total_obj > budget) break;

                            insert_exact_max_frontier_point_(
                                parent_feature.upper,
                                total_obj,
                                rp.value
                            );
                        }
                    }
                }

                prior_min_feasible_obj =
                    std::min(prior_min_feasible_obj, split_min_obj);
                saw_solution = true;
            }
        }

        if (!saw_solution) {
            return nullptr;
        }

        auto entry =
            std::make_shared<ExactGlobalImportanceFrontierCacheEntry_>();

        entry->solved_budget = budget;
        entry->frontiers.min_feasible_obj = prior_min_feasible_obj;
        entry->frontiers.features.reserve(feature_acc.size());

        for (const auto& [variable, acc] : feature_acc) {
            if (acc.lower.empty() || acc.upper.empty()) {
                throw std::runtime_error(
                    "Sparse cached global importance created an empty "
                    "feature frontier."
                );
            }

            ExactSparseFeatureImportanceFrontiers_ feature;
            feature.variable = variable;
            feature.lower = exact_frontier_map_to_vector_(acc.lower);
            feature.upper = exact_frontier_map_to_vector_(acc.upper);
            entry->frontiers.features.push_back(std::move(feature));
        }

        cache[cache_key] = entry;
        return entry;
    }


public:

    uint8_t reachable_prediction_mask_for_training_sample(int sample_idx) const {
        if (!result) {
            throw std::runtime_error(
                "No Rashomon trie has been constructed. Call fit(..., rashomon_mode=true) first."
            );
        }
        if (sample_idx < 0 || sample_idx >= n_samples) {
            throw std::runtime_error("sample_idx is out of range.");
        }

        return reachable_prediction_mask_for_training_sample_rec_(
            result,
            sample_idx
        );
    }

    ReachableActions
    reachable_actions_for_training_sample(
        int sample_idx
    ) const {
        if (!result) {
            throw std::runtime_error(
                "No Rashomon trie has been constructed. "
                "Call fit(..., rashomon_mode=true) first."
            );
        }

        if (
            sample_idx < 0 ||
            sample_idx >= n_samples
        ) {
            throw std::runtime_error(
                "sample_idx is out of range."
            );
        }

        return reachable_actions_for_training_sample_rec_(
            result,
            sample_idx
        );
    }

    bool training_sample_has_multiple_reachable_predictions(
        int sample_idx
    ) const {
        const ReachableActions actions =
            reachable_actions_for_training_sample(
                sample_idx
            );

        return popcnt64(
            actions.class_mask
        ) >= 2;
    }

    bool training_sample_can_defer(
        int sample_idx
    ) const {
        return reachable_actions_for_training_sample(
            sample_idx
        ).can_defer;
    }

    std::vector<int> training_samples_with_multiple_reachable_predictions() const {
        if (!result) {
            throw std::runtime_error(
                "No Rashomon trie has been constructed. Call fit(..., rashomon_mode=true) first."
            );
        }

        std::vector<int> out;

        for (int i = 0; i < n_samples; ++i) {
            if (training_sample_has_multiple_reachable_predictions(i)) {
                out.push_back(i);
            }
        }

        return out;
    }

    // main entry: enumerate ALL valid trees under the trie root (or budget_override if >=0),
    // returning (training_objective, prediction vector on evaluation dataset) for each tree.
    // NOTE: this can be extremely large in memory if the Rashomon set is huge.
    std::vector<PredPackWithObj> get_all_predictions_packed_trie(const std::vector<std::vector<uint8_t>>& X_row_major, int budget_override = -1) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features); // get the evaluation dataset in column major

        // decide budget
        int budget = (budget_override >= 0) ? budget_override : result->budget;

        // root eval mask = all eval rows are 1, with the padding 0s (eval_rows % 64)
        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        // collect grouped by objective
        auto buckets = collect_preds_by_obj_(result.get(), budget, root_mask, ctx);

        // flatten
        std::vector<PredPackWithObj> out;
        // compute total count for reserve
        size_t total = 0;
        for (const auto& b : buckets) total += b.preds.size();
        out.reserve(total);

        for (auto &b : buckets) {
            for (auto &p : b.preds) {
                out.push_back(PredPackWithObj{b.obj, std::move(p)});
            }
        }
        return out;
    }

    std::vector<DeferralsWithObj> get_all_deferrals_objs_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        int budget_override = -1
    ) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features);

        const int budget = (budget_override >= 0) ? budget_override : result->budget;

        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        auto buckets = collect_deferrals_by_obj_(
            result.get(),
            budget,
            root_mask,
            ctx
        );

        std::vector<DeferralsWithObj> out;

        size_t total = 0;
        for (const auto& b : buckets) {
            total += b.deferrals.size();
        }
        out.reserve(total);

        for (auto& b : buckets) {
            for (int d : b.deferrals) {
                out.push_back(DeferralsWithObj{b.obj, d});
            }
        }

        return out;
    }

    std::vector<int> get_all_deferrals_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        int budget_override = -1
    ) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features);

        const int budget = (budget_override >= 0) ? budget_override : result->budget;

        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        auto buckets = collect_deferrals_by_obj_(
            result.get(),
            budget,
            root_mask,
            ctx
        );

        std::vector<int> out;

        size_t total = 0;
        for (const auto& b : buckets) {
            total += b.deferrals.size();
        }
        out.reserve(total);

        for (auto& b : buckets) {
            for (int d : b.deferrals) {
                out.push_back(d);
            }
        }

        return out;
    }

    std::vector<MistakesWithObj> get_all_misclassifications_objs_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& y_eval,
        int budget_override = -1,
        const std::vector<int>& bb_pred_eval = {}
    ) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features);

        if ((int)y_eval.size() != ctx.n_eval) {
            throw std::runtime_error("Eval y has different number of rows than Eval X.");
        }

        const int budget = (budget_override >= 0) ? budget_override : result->budget;

        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        std::vector<Packed> Y_eval_bits = build_eval_y_bits_(
            y_eval,
            num_classes,
            ctx.n_words,
            ctx.tail_mask
        );

        Packed BBwrong_eval_storage((size_t)ctx.n_words);
        const Packed* BBwrong_eval_ptr = nullptr;

        if (use_deferral) {
            if (bb_pred_eval.empty()) {
                throw std::runtime_error(
                    "get_all_misclassifications_objs_packed_trie: "
                    "deferral was enabled during fit, so bb_pred_eval is required."
                );
            }

            BBwrong_eval_storage = build_eval_bb_wrong_bits_(
                y_eval,
                bb_pred_eval,
                num_classes,
                ctx.n_words,
                ctx.tail_mask
            );

            BBwrong_eval_ptr = &BBwrong_eval_storage;
        }



       auto buckets = collect_mistakes_by_obj_(
            result.get(),
            budget,
            root_mask,
            ctx,
            Y_eval_bits,
            BBwrong_eval_ptr
        );

        std::vector<MistakesWithObj> out;

        size_t total = 0;
        for (const auto& b : buckets) {
            total += b.mistakes.size();
        }
        out.reserve(total);

        for (auto& b : buckets) {
            for (int m : b.mistakes) {
                out.push_back(MistakesWithObj{b.obj, m});
            }
        }

        return out;
    }

    std::vector<int> get_all_misclassifications_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& y_eval,
        int budget_override = -1,
        const std::vector<int>& bb_pred_eval = {}
    ) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features);

        if ((int)y_eval.size() != ctx.n_eval) {
            throw std::runtime_error("Eval y has different number of rows than Eval X.");
        }

        const int budget = (budget_override >= 0) ? budget_override : result->budget;

        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        std::vector<Packed> Y_eval_bits = build_eval_y_bits_(
            y_eval,
            num_classes,
            ctx.n_words,
            ctx.tail_mask
        );

        Packed BBwrong_eval_storage((size_t)ctx.n_words);
        const Packed* BBwrong_eval_ptr = nullptr;

        if (use_deferral) {
            if (bb_pred_eval.empty()) {
                throw std::runtime_error(
                    "get_all_misclassifications_objs_packed_trie: "
                    "deferral was enabled during fit, so bb_pred_eval is required."
                );
            }

            BBwrong_eval_storage = build_eval_bb_wrong_bits_(
                y_eval,
                bb_pred_eval,
                num_classes,
                ctx.n_words,
                ctx.tail_mask
            );

            BBwrong_eval_ptr = &BBwrong_eval_storage;
        }

        auto buckets = collect_mistakes_by_obj_(
            result.get(),
            budget,
            root_mask,
            ctx,
            Y_eval_bits,
            BBwrong_eval_ptr
        );

        std::vector<int> out;

        size_t total = 0;
        for (const auto& b : buckets) {
            total += b.mistakes.size();
        }
        out.reserve(total);

        for (auto& b : buckets) {
            for (int m : b.mistakes) {
                out.push_back(m);
            }
        }

        return out;
    }


    std::vector<ExactImportanceInterval>
    get_exact_replacement_importance_intervals_cached_frontier_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& y_eval,
        int budget_override = -1,
        const std::vector<std::vector<int>>& variable_columns_in = {},
        const std::vector<int>& bb_pred_eval = {},
        const std::vector<std::vector<int>>&
            matched_group_of_row_by_variable_eval = {},
        const std::vector<std::vector<int>>&
            matched_group_size_by_variable_eval = {}
    ) const {
        auto setup =
            prepare_exact_replacement_interval_eval_(
                X_row_major,
                y_eval,
                budget_override,
                variable_columns_in,
                bb_pred_eval,
                matched_group_of_row_by_variable_eval,
                matched_group_size_by_variable_eval
            );

        if (setup.ctx.n_eval <= 0) {
            return {};
        }

        require_exact_importance_eval_is_training_(setup, y_eval);

        if (!result) {
            throw std::runtime_error(
                "No Rashomon trie has been constructed. Call fit() first."
            );
        }

        if (setup.budget > result->budget) {
            throw std::runtime_error(
                "Cached global importance can only query a budget no larger "
                "than the fitted root graph budget."
            );
        }

        const int delta = result->budget - setup.budget;

        const int number_of_variables =
            static_cast<int>(setup.variable_columns.size());

        const Packed* BBwrong_eval_ptr =
            setup.has_bb_wrong
                ? &setup.bb_wrong
                : nullptr;

        const std::vector<std::vector<int>>*
            matched_group_of_row_by_variable_eval_ptr =
                setup.use_matched_groups
                    ? &matched_group_of_row_by_variable_eval
                    : nullptr;

        const std::vector<std::vector<double>>*
            matched_group_inv_size_by_variable_eval_ptr =
                setup.use_matched_groups
                    ? &setup.matched_group_inv_sizes
                    : nullptr;

        const std::vector<uint8_t>*
            matched_group_effectively_uniform_by_variable_eval_ptr =
                setup.use_matched_groups
                    ? &setup.matched_group_effectively_uniform
                    : nullptr;

        ExactMatchedScratch_ matched_scratch;
        ExactMatchedScratch_* matched_scratch_ptr =
            setup.use_matched_groups
                ? &matched_scratch
                : nullptr;

        // both are intentionally empty at the root. we only materialize replacement states and
        // semantic path constraints when a feature is actually encountered on a feasible split path.
        ExactSparseReplacementStates_ root_states;
        ExactImportanceSemanticPath_ root_path;

        ExactGlobalImportanceFrontierCache_ cache;
        cache.reserve(1024);

        auto root_entry =
            collect_exact_global_importance_frontiers_cached_(
                result.get(),
                static_cast<int>(trained_depth_budget),
                delta,
                setup.root_mask,
                setup.root_mask,
                root_states,
                root_path,
                setup.internal_to_variable,
                setup.ctx,
                setup.y_bits,
                BBwrong_eval_ptr,
                matched_group_of_row_by_variable_eval_ptr,
                matched_group_inv_size_by_variable_eval_ptr,
                matched_group_effectively_uniform_by_variable_eval_ptr,
                matched_scratch_ptr,
                cache
            );

        if (!root_entry) {
            return {};
        }

        const auto& root_frontiers = root_entry->frontiers;

        if (
            root_frontiers.min_feasible_obj ==
                std::numeric_limits<int>::max() ||
            root_frontiers.min_feasible_obj > setup.budget
        ) {
            return {};
        }

        std::vector<ExactImportanceInterval> out(
            static_cast<std::size_t>(number_of_variables),
            {0.0, 0.0}
        );

        const double inv_n =
            1.0 / static_cast<double>(setup.ctx.n_eval);

        for (int variable = 0;
             variable < number_of_variables;
             ++variable) {

            const auto* feature =
                find_exact_sparse_feature_frontiers_(
                    root_frontiers,
                    variable
                );

            // absent from the sparse root result means no feasible tree in
            // the queried Rashomon set ever uses this replacement variable,
            // so its importance is exactly zero.
            if (!feature) {
                out[static_cast<std::size_t>(variable)] = {0.0, 0.0};
                continue;
            }

            const double lower =
                exact_frontier_value_at_budget_(
                    feature->lower,
                    setup.budget
                );

            const double upper =
                exact_frontier_value_at_budget_(
                    feature->upper,
                    setup.budget
                );

            out[static_cast<std::size_t>(variable)] = {
                lower * inv_n,
                upper * inv_n
            };
        }

        return out;
    }

    // if sum_samplewise_extrema=false, returns
    //   [ min_f Phi_j(f), max_f Phi_j(f) ]
    // if sum_samplewise_extrema=true, returns a wider range
    //   [ (1/n) sum_i min_f phi_{ij}(f),
    //     (1/n) sum_i max_f phi_{ij}(f) ]
    std::vector<ExactImportanceInterval>
    get_exact_replacement_importance_intervals_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& y_eval,
        int budget_override = -1,
        const std::vector<std::vector<int>>& variable_columns_in = {},
        const std::vector<int>& bb_pred_eval = {},
        const std::vector<std::vector<int>>&
            matched_group_of_row_by_variable_eval = {},
        const std::vector<std::vector<int>>&
            matched_group_size_by_variable_eval = {},
        bool sum_samplewise_extrema = false
    ) const {
        auto setup =
            prepare_exact_replacement_interval_eval_(
                X_row_major,
                y_eval,
                budget_override,
                variable_columns_in,
                bb_pred_eval,
                matched_group_of_row_by_variable_eval,
                matched_group_size_by_variable_eval
            );

        if (setup.ctx.n_eval <= 0) {
            return {};
        }

        const int number_of_variables =
            static_cast<int>(setup.variable_columns.size());

        std::vector<ExactImportanceInterval> out(
            static_cast<std::size_t>(number_of_variables),
            {0.0, 0.0}
        );

        const Packed* BBwrong_eval_ptr =
            setup.has_bb_wrong
                ? &setup.bb_wrong
                : nullptr;

        const std::vector<std::vector<int>>*
            matched_group_of_row_by_variable_eval_ptr =
                setup.use_matched_groups
                    ? &matched_group_of_row_by_variable_eval
                    : nullptr;

        const std::vector<std::vector<double>>*
            matched_group_inv_size_by_variable_eval_ptr =
                setup.use_matched_groups
                    ? &setup.matched_group_inv_sizes
                    : nullptr;

        const std::vector<uint8_t>*
            matched_group_effectively_uniform_by_variable_eval_ptr =
                setup.use_matched_groups
                    ? &setup.matched_group_effectively_uniform
                    : nullptr;

        if (!sum_samplewise_extrema) {
            std::vector<ExactReplacementState_> root_states(
                static_cast<std::size_t>(number_of_variables)
            );

            ExactMatchedScratch_ matched_scratch;
            ExactMatchedScratch_* matched_scratch_ptr =
                setup.use_matched_groups
                    ? &matched_scratch
                    : nullptr;

            auto buckets =
                collect_exact_global_importance_extrema_by_obj_(
                    result.get(),
                    setup.budget,
                    setup.root_mask,
                    setup.root_mask,
                    root_states,
                    setup.internal_to_variable,
                    setup.ctx,
                    setup.y_bits,
                    BBwrong_eval_ptr,
                    matched_group_of_row_by_variable_eval_ptr,
                    matched_group_inv_size_by_variable_eval_ptr,
                    matched_group_effectively_uniform_by_variable_eval_ptr,
                    matched_scratch_ptr
                );

            if (buckets.empty()) {
                return {};
            }

            std::vector<double> lower(
                static_cast<std::size_t>(number_of_variables),
                std::numeric_limits<double>::infinity()
            );
            std::vector<double> upper(
                static_cast<std::size_t>(number_of_variables),
                -std::numeric_limits<double>::infinity()
            );

            for (const auto& bucket : buckets) {
                for (int variable = 0;
                     variable < number_of_variables;
                     ++variable) {

                    const std::size_t j =
                        static_cast<std::size_t>(variable);

                    lower[j] =
                        std::min(lower[j], bucket.extrema.lower[j]);
                    upper[j] =
                        std::max(upper[j], bucket.extrema.upper[j]);
                }
            }

            const double inv_n =
                1.0 / static_cast<double>(setup.ctx.n_eval);

            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {

                const std::size_t j =
                    static_cast<std::size_t>(variable);

                out[j] = {
                    lower[j] * inv_n,
                    upper[j] * inv_n
                };
            }

            return out;
        }
        
        const double inv_n =
            1.0 / static_cast<double>(setup.ctx.n_eval);

        ExactNodeVariableMasks_ node_variable_masks;
        build_exact_node_variable_masks_(
            result.get(),
            setup.internal_to_variable,
            number_of_variables,
            node_variable_masks
        );


        static constexpr bool SAMPLEWISE_ALL_FEATURES_AT_ONCE = true;

        if (SAMPLEWISE_ALL_FEATURES_AT_ONCE) {
            std::vector<ExactReplacementState_> root_states(
                static_cast<std::size_t>(number_of_variables)
            );
            std::vector<uint8_t> requested_variables(
                static_cast<std::size_t>(number_of_variables),
                1
            );
            ExactMatchedScratch_ all_matched_scratch;

            auto all_extrema =
                collect_exact_local_importance_numerator_extrema_all_at_most_(
                result.get(),
                setup.budget,
                setup.root_mask,
                setup.root_mask,
                root_states,
                requested_variables,
                setup.internal_to_variable,
                setup.ctx,
                setup.y_bits,
                BBwrong_eval_ptr,
                setup.use_matched_groups
                    ? &matched_group_of_row_by_variable_eval
                    : nullptr,
                setup.use_matched_groups
                    ? &matched_group_size_by_variable_eval
                    : nullptr,
                setup.use_matched_groups
                    ? &setup.matched_group_effectively_uniform
                    : nullptr,
                setup.use_matched_groups
                    ? &all_matched_scratch
                    : nullptr,
                node_variable_masks
            );

        if (all_extrema.empty()) return {};

        for (int variable = 0;
             variable < number_of_variables;
             ++variable) {
            const std::size_t j = static_cast<std::size_t>(variable);
            auto& extrema = all_extrema[j];
            if (extrema.empty()) return {};
            if (extrema.all_zero) {
                out[j] = {0.0, 0.0};
                continue;
            }

            const bool matched_effectively_uniform =
                setup.use_matched_groups &&
                setup.matched_group_effectively_uniform[j] != 0;

            if (!setup.use_matched_groups || matched_effectively_uniform) {
                if (extrema.is_point()) {
                    const int64_t point_sum = fast_sum_i32_(
                        extrema.point.data(),
                        static_cast<std::size_t>(setup.ctx.n_eval)
                    );
                    const double scale =
                        inv_n / static_cast<double>(setup.ctx.n_eval);
                    const double value =
                        static_cast<double>(point_sum) * scale;
                    out[j] = {value, value};
                } else {
                    const int64_t lower_sum = fast_sum_i32_(
                        extrema.lower.data(),
                        static_cast<std::size_t>(setup.ctx.n_eval)
                    );
                    const int64_t upper_sum = fast_sum_i32_(
                        extrema.upper.data(),
                        static_cast<std::size_t>(setup.ctx.n_eval)
                    );
                    const double scale =
                        inv_n / static_cast<double>(setup.ctx.n_eval);
                    out[j] = {
                        static_cast<double>(lower_sum) * scale,
                        static_cast<double>(upper_sum) * scale
                    };
                }
                continue;
            }

            const auto& group_of_row =
                matched_group_of_row_by_variable_eval[j];
            const auto& group_sizes =
                matched_group_size_by_variable_eval[j];
            double lower_sum = 0.0;
            double upper_sum = 0.0;
            for (int row = 0; row < setup.ctx.n_eval; ++row) {
                const int group = group_of_row[static_cast<std::size_t>(row)];
                const int group_size = group_sizes[static_cast<std::size_t>(group)];
                if (extrema.is_point()) {
                    const double value =
                        static_cast<double>(
                            extrema.point[static_cast<std::size_t>(row)]
                        ) / static_cast<double>(group_size);
                    lower_sum += value;
                    upper_sum += value;
                } else {
                    lower_sum +=
                        static_cast<double>(
                            extrema.lower[static_cast<std::size_t>(row)]
                        ) / static_cast<double>(group_size);
                    upper_sum +=
                        static_cast<double>(
                            extrema.upper[static_cast<std::size_t>(row)]
                        ) / static_cast<double>(group_size);
                }
            }
            out[j] = {lower_sum * inv_n, upper_sum * inv_n};
        }

            return out;
        }

        for (int variable = 0;
             variable < number_of_variables;
             ++variable) {

            ExactReplacementState_ root_state;
            ExactMatchedScratch_ matched_scratch;

            const std::vector<int>* group_of_row_ptr =
                setup.use_matched_groups
                    ? &matched_group_of_row_by_variable_eval[
                        static_cast<std::size_t>(variable)
                    ]
                    : nullptr;

            const std::vector<int>* group_size_ptr =
                setup.use_matched_groups
                    ? &matched_group_size_by_variable_eval[
                        static_cast<std::size_t>(variable)
                    ]
                    : nullptr;

            const bool matched_effectively_uniform =
                setup.use_matched_groups &&
                setup.matched_group_effectively_uniform[
                    static_cast<std::size_t>(variable)
                ] != 0;

            auto extrema =
                collect_exact_local_importance_numerator_extrema_at_most_(
                    result.get(),
                    setup.budget,
                    setup.root_mask,
                    setup.root_mask,
                    root_state,
                    variable,
                    setup.internal_to_variable,
                    setup.ctx,
                    setup.y_bits,
                    BBwrong_eval_ptr,
                    group_of_row_ptr,
                    group_size_ptr,
                    matched_effectively_uniform,
                    setup.use_matched_groups
                        ? &matched_scratch
                        : nullptr,
                    node_variable_masks
                );

            if (extrema.empty()) {
                return {};
            }

            if (extrema.all_zero) {
                out[static_cast<std::size_t>(variable)] = {0.0, 0.0};
                continue;
            }

            if (!setup.use_matched_groups || matched_effectively_uniform) {
                if (extrema.is_point()) {
                    const int64_t point_sum =
                        fast_sum_i32_(
                            extrema.point.data(),
                            static_cast<std::size_t>(setup.ctx.n_eval)
                        );
                    const double scale =
                        inv_n / static_cast<double>(setup.ctx.n_eval);
                    const double value =
                        static_cast<double>(point_sum) * scale;
                    out[static_cast<std::size_t>(variable)] = {value, value};
                    continue;
                }

                const int64_t lower_sum =
                    fast_sum_i32_(
                        extrema.lower.data(),
                        static_cast<std::size_t>(setup.ctx.n_eval)
                    );

                const int64_t upper_sum =
                    fast_sum_i32_(
                        extrema.upper.data(),
                        static_cast<std::size_t>(setup.ctx.n_eval)
                    );

                const double scale =
                    inv_n / static_cast<double>(setup.ctx.n_eval);

                out[static_cast<std::size_t>(variable)] = {
                    static_cast<double>(lower_sum) * scale,
                    static_cast<double>(upper_sum) * scale
                };
                continue;
            }

            double lower_sum = 0.0;
            double upper_sum = 0.0;

            for (int row = 0; row < setup.ctx.n_eval; ++row) {
                const int group =
                    (*group_of_row_ptr)[
                        static_cast<std::size_t>(row)
                    ];
                const int group_size =
                    (*group_size_ptr)[
                        static_cast<std::size_t>(group)
                    ];

                if (extrema.is_point()) {
                    const double value =
                        static_cast<double>(
                            extrema.point[static_cast<std::size_t>(row)]
                        ) /
                        static_cast<double>(group_size);
                    lower_sum += value;
                    upper_sum += value;
                } else {
                    lower_sum +=
                        static_cast<double>(
                            extrema.lower[static_cast<std::size_t>(row)]
                        ) /
                        static_cast<double>(group_size);

                    upper_sum +=
                        static_cast<double>(
                            extrema.upper[static_cast<std::size_t>(row)]
                        ) /
                        static_cast<double>(group_size);
                }
            }

            out[static_cast<std::size_t>(variable)] = {
                lower_sum * inv_n,
                upper_sum * inv_n
            };
        }

        return out;
    }


    std::vector<ExactReplacementMistakesWithObj>
    get_all_exact_replacement_misclassifications_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& y_eval,
        int budget_override = -1,

        // optional original-variable grouping.
        // variable_columns[j] = {threshold_col_0, threshold_col_1, ...}
        const std::vector<std::vector<int>>& variable_columns_in = {},

        const std::vector<int>& bb_pred_eval = {},

        // matched_group_of_row_by_variable_eval[j][i]
        // gives the matched-group ID of evaluation row i
        // for replacement variable j.
        const std::vector<std::vector<int>>&
            matched_group_of_row_by_variable_eval = {},

        // matched_group_size_by_variable_eval[j][g]
        // gives the number of evaluation rows in group g
        // for replacement variable j. zero-sized groups are allowed because
        // an original group may be absent from a bootstrap.
        const std::vector<std::vector<int>>&
            matched_group_size_by_variable_eval = {}
    ) const {
        if (!result) {
            throw std::runtime_error(
                "No Rashomon trie has been constructed. Call fit() first."
            );
        }

        EvalCtx ctx =
            build_eval_ctx_(
                X_row_major,
                this->n_features
            );

        if ((int)y_eval.size() != ctx.n_eval) {
            throw std::runtime_error(
                "Eval y has different number of rows than Eval X."
            );
        }

        if (ctx.n_eval <= 0) {
            return {};
        }

        const int budget =
            (budget_override >= 0)
                ? budget_override
                : result->budget;

        std::vector<std::vector<int>> variable_columns;

        if (!variable_columns_in.empty()) {
            variable_columns = variable_columns_in;
        } else {
            const int first_cont =
                first_continuous_feature_();

            for (int f = 0; f < first_cont; ++f) {
                variable_columns.push_back({f});
            }

            for (int g = 0;
                 g < (int)continuous_starts.size();
                 ++g) {

                const int start =
                    continuous_starts[(size_t)g];

                const int end =
                    continuous_group_end_(g);

                std::vector<int> cols;
                cols.reserve((size_t)(end - start));

                for (int f = start; f < end; ++f) {
                    cols.push_back(f);
                }

                variable_columns.push_back(std::move(cols));
            }
        }

        const int number_of_variables =
            (int)variable_columns.size();

        const bool has_group_of_row =
            !matched_group_of_row_by_variable_eval.empty();

        const bool has_group_sizes =
            !matched_group_size_by_variable_eval.empty();

        if (has_group_of_row != has_group_sizes) {
            throw std::runtime_error(
                "Matched-group evaluation requires both "
                "group-of-row and group-size arrays."
            );
        }

        const bool use_matched_groups =
            has_group_of_row;

        std::vector<std::vector<double>>
            matched_group_inv_size_by_variable_eval;

        std::vector<uint8_t>
            matched_group_effectively_uniform_by_variable_eval;

        if (use_matched_groups) {
            if (
                (int)matched_group_of_row_by_variable_eval.size() !=
                    number_of_variables ||
                (int)matched_group_size_by_variable_eval.size() !=
                    number_of_variables
            ) {
                throw std::runtime_error(
                    "Matched-group arrays must contain exactly one "
                    "entry per original variable."
                );
            }

            matched_group_inv_size_by_variable_eval.resize(
                (std::size_t)number_of_variables
            );

            matched_group_effectively_uniform_by_variable_eval.assign(
                (std::size_t)number_of_variables,
                0
            );

            for (int variable = 0;
                 variable < number_of_variables;
                 ++variable) {

                const auto& group_of_row =
                    matched_group_of_row_by_variable_eval[
                        (std::size_t)variable
                    ];

                const auto& group_sizes =
                    matched_group_size_by_variable_eval[
                        (std::size_t)variable
                    ];

                if ((int)group_of_row.size() != ctx.n_eval) {
                    throw std::runtime_error(
                        "Matched-group row map has the wrong "
                        "number of evaluation rows."
                    );
                }

                if (group_sizes.empty()) {
                    throw std::runtime_error(
                        "Matched-group size array is empty."
                    );
                }

                std::vector<int> observed_group_sizes(
                    group_sizes.size(),
                    0
                );

                for (int row = 0;
                     row < ctx.n_eval;
                     ++row) {

                    const int group =
                        group_of_row[(std::size_t)row];

                    if (
                        group < 0 ||
                        group >= (int)group_sizes.size()
                    ) {
                        throw std::runtime_error(
                            "Matched-group row map contains an "
                            "out-of-range group ID."
                        );
                    }

                    ++observed_group_sizes[
                        (std::size_t)group
                    ];
                }

                auto& inverse_group_sizes =
                    matched_group_inv_size_by_variable_eval[
                        (std::size_t)variable
                    ];

                inverse_group_sizes.assign(
                    group_sizes.size(),
                    0.0
                );

                int number_of_nonempty_groups = 0;

                for (std::size_t group = 0;
                     group < group_sizes.size();
                     ++group) {

                    if (group_sizes[group] < 0) {
                        throw std::runtime_error(
                            "Matched-group size cannot be negative."
                        );
                    }

                    if (
                        observed_group_sizes[group] !=
                        group_sizes[group]
                    ) {
                        throw std::runtime_error(
                            "Matched-group size array does not agree "
                            "with the row-to-group map."
                        );
                    }

                    if (group_sizes[group] > 0) {
                        inverse_group_sizes[group] =
                            1.0 /
                            static_cast<double>(
                                group_sizes[group]
                            );

                        ++number_of_nonempty_groups;
                    }
                }

                if (number_of_nonempty_groups <= 0) {
                    throw std::runtime_error(
                        "Matched-group partition has no rows."
                    );
                }

                matched_group_effectively_uniform_by_variable_eval[
                    (std::size_t)variable
                ] =
                    number_of_nonempty_groups == 1
                        ? 1
                        : 0;
            }
        }

        std::vector<int> internal_to_variable(
            (size_t)this->n_features,
            -1
        );

        for (int variable = 0;
             variable < number_of_variables;
             ++variable) {

            const auto& cols =
                variable_columns[(size_t)variable];

            if (cols.empty()) {
                throw std::runtime_error(
                    "variable_columns contains an empty variable."
                );
            }

            for (int f : cols) {
                if (f < 0 || f >= this->n_features) {
                    throw std::runtime_error(
                        "variable_columns contains an out-of-range internal column."
                    );
                }

                if (internal_to_variable[(size_t)f] != -1) {
                    throw std::runtime_error(
                        "An internal column appears in more than one variable."
                    );
                }

                internal_to_variable[(size_t)f] =
                    variable;
            }
        }

        for (int f = 0; f < this->n_features; ++f) {
            if (internal_to_variable[(size_t)f] < 0) {
                throw std::runtime_error(
                    "Every internal feature column must belong to exactly one variable."
                );
            }
        }

        Packed root_mask =
            eval_root_mask_(
                ctx.n_words,
                ctx.tail_mask
            );

        std::vector<Packed> Y_eval_bits =
            build_eval_y_bits_(
                y_eval,
                num_classes,
                ctx.n_words,
                ctx.tail_mask
            );

        Packed BBwrong_eval_storage((size_t)ctx.n_words);
        const Packed* BBwrong_eval_ptr = nullptr;

        if (use_deferral) {
            if (bb_pred_eval.empty()) {
                throw std::runtime_error(
                    "Deferral was enabled during fit, so bb_pred_eval is required."
                );
            }

            BBwrong_eval_storage =
                build_eval_bb_wrong_bits_(
                    y_eval,
                    bb_pred_eval,
                    num_classes,
                    ctx.n_words,
                    ctx.tail_mask
                );

            BBwrong_eval_ptr =
                &BBwrong_eval_storage;
        }

        // lazy root state: no replacement variable has appeared yet, so no
        // target/donor masks need to be materialized.
        std::vector<ExactReplacementState_> root_states(
            (size_t)number_of_variables
        );

        const std::vector<std::vector<int>>*
            matched_group_of_row_by_variable_eval_ptr =
                use_matched_groups
                    ? &matched_group_of_row_by_variable_eval
                    : nullptr;

        const std::vector<std::vector<double>>*
            matched_group_inv_size_by_variable_eval_ptr =
                use_matched_groups
                    ? &matched_group_inv_size_by_variable_eval
                    : nullptr;

        const std::vector<uint8_t>*
            matched_group_effectively_uniform_by_variable_eval_ptr =
                use_matched_groups
                    ? &matched_group_effectively_uniform_by_variable_eval
                    : nullptr;

        ExactMatchedScratch_ matched_scratch;

        ExactMatchedScratch_* matched_scratch_ptr =
            use_matched_groups
                ? &matched_scratch
                : nullptr;

        auto buckets =
            collect_exact_replacement_mistakes_by_obj_(
                result.get(),
                budget,
                root_mask,
                root_mask,
                root_states,
                internal_to_variable,
                ctx,
                Y_eval_bits,
                BBwrong_eval_ptr,
                matched_group_of_row_by_variable_eval_ptr,
                matched_group_inv_size_by_variable_eval_ptr,
                matched_group_effectively_uniform_by_variable_eval_ptr,
                matched_scratch_ptr
            );

        std::vector<ExactReplacementMistakesWithObj> out;

        size_t total = 0;
        for (const auto& b : buckets) {
            total += b.counts.size();
        }

        out.reserve(total);

        for (auto& b : buckets) {
            for (auto& counts : b.counts) {
                ExactReplacementMistakesWithObj row;
                row.obj = b.obj;

                row.mistakes.resize(
                    (std::size_t)number_of_variables + 1,
                    0.0
                );

                row.mistakes[0] =
                    (double)counts.original_mistakes;

                for (int variable = 0;
                     variable < number_of_variables;
                     ++variable) {

                    row.mistakes[
                        (std::size_t)variable + 1
                    ] =
                        counts.replacement_expected_mistakes[
                            (std::size_t)variable
                        ];
                }

                out.push_back(std::move(row));
            }
        }

        return out;
    }


};