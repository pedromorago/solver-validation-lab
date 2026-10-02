/*
 * Exact preflop all-in equity, heads-up, for the 169 starting-hand classes.
 *
 *   preflop_equity table OUT       every class against every class, written to OUT
 *   preflop_equity matchup H1 H2 [BOARD]
 *                                  one matchup, e.g. AhKh QsQd or AhKh QsQd Jh7c2d
 *
 * Equity is counted in integer units, the same way as svlab.equity.exact_equity
 * with two players: each runout is worth 2 units, 2 to the winner or 1 each on
 * a split. `matchup` prints the units of both hands so the Python engine can
 * check this program exactly.
 *
 * The table enumerates every pair of hands that share no card (812,175 of
 * them), groups the pairs that are the same matchup with suits relabelled,
 * enumerates all 1,712,304 boards once per group and adds each pair's units to
 * its two classes. Output: one line per (i, j) with the number of hand pairs
 * and the units won by class i, so that equity(i, j) = units / (pairs * boards * 2).
 *
 * Build: gcc -O3 -march=native -fopenmp -o preflop_equity preflop_equity.c
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NCLASS 169
static const char RANKS[] = "23456789TJQKA";
static const char SUITS[] = "cdhs";

static int card_rank(int c) { return c >> 2; }
static int card_suit(int c) { return c & 3; }

static int parse_card(const char *s) {
    const char *r = strchr(RANKS, s[0]);
    const char *u = strchr(SUITS, s[1]);
    if (!r || !u || !s[0] || !s[1]) return -1;
    return (int)(r - RANKS) * 4 + (int)(u - SUITS);
}

/* Highest card of the best straight in a 13-bit rank mask, or -1. The wheel is five-high (3). */
static int straight_high(unsigned mask) {
    for (int high = 12; high >= 4; high--) {
        unsigned need = 0x1Fu << (high - 4);
        if ((mask & need) == need) return high;
    }
    if ((mask & 0x100Fu) == 0x100Fu) return 3;
    return -1;
}

/* Packs a category and up to five ranks so that a bigger value is a better hand,
 * matching the tuples of svlab.evaluator.evaluate. */
static uint32_t pack(int cat, const int *k, int n) {
    uint32_t v = (uint32_t)cat << 20;
    for (int i = 0; i < n; i++) v |= (uint32_t)k[i] << (16 - 4 * i);
    return v;
}

static uint32_t eval7(const int *cards) {
    int counts[13] = {0};
    unsigned suit_mask[4] = {0}, rank_mask = 0;
    for (int i = 0; i < 7; i++) {
        int r = card_rank(cards[i]), s = card_suit(cards[i]);
        counts[r]++;
        suit_mask[s] |= 1u << r;
        rank_mask |= 1u << r;
    }
    int k[5] = {0};
    for (int s = 0; s < 4; s++) {
        if (__builtin_popcount(suit_mask[s]) >= 5) {
            int high = straight_high(suit_mask[s]);
            if (high >= 0) { k[0] = high; return pack(8, k, 1); }
        }
    }
    int quad = -1, trips[2] = {-1, -1}, nt = 0, pairs[3] = {-1, -1, -1}, np = 0;
    for (int r = 12; r >= 0; r--) {
        if (counts[r] == 4) quad = r;
        else if (counts[r] == 3) trips[nt++] = r;
        else if (counts[r] == 2) pairs[np++] = r;
    }
    if (quad >= 0) {
        int kicker = -1;
        for (int r = 12; r >= 0; r--) if (counts[r] && r != quad) { kicker = r; break; }
        k[0] = quad; k[1] = kicker;
        return pack(7, k, 2);
    }
    if (nt >= 1) {
        int best_pair = nt == 2 ? trips[1] : -1;
        if (np > 0 && pairs[0] > best_pair) best_pair = pairs[0];
        if (best_pair >= 0) { k[0] = trips[0]; k[1] = best_pair; return pack(6, k, 2); }
    }
    for (int s = 0; s < 4; s++) {
        if (__builtin_popcount(suit_mask[s]) >= 5) {
            int n = 0;
            for (int r = 12; r >= 0 && n < 5; r--) if (suit_mask[s] >> r & 1) k[n++] = r;
            return pack(5, k, 5);
        }
    }
    int high = straight_high(rank_mask);
    if (high >= 0) { k[0] = high; return pack(4, k, 1); }
    if (nt == 1) {
        int n = 1; k[0] = trips[0];
        for (int r = 12; r >= 0 && n < 3; r--) if (counts[r] && r != trips[0]) k[n++] = r;
        return pack(3, k, 3);
    }
    if (np >= 2) {
        k[0] = pairs[0]; k[1] = pairs[1];
        for (int r = 12; r >= 0; r--) if (counts[r] && r != pairs[0] && r != pairs[1]) { k[2] = r; break; }
        return pack(2, k, 3);
    }
    if (np == 1) {
        int n = 1; k[0] = pairs[0];
        for (int r = 12; r >= 0 && n < 4; r--) if (counts[r] && r != pairs[0]) k[n++] = r;
        return pack(1, k, 4);
    }
    int n = 0;
    for (int r = 12; r >= 0 && n < 5; r--) if (counts[r]) k[n++] = r;
    return pack(0, k, 5);
}

/* Units won by hand a and hand b over every completion of `board` (0 to 5 cards). */
static void matchup_units(const int *a, const int *b, const int *board, int nboard,
                          uint64_t *ua, uint64_t *ub, uint64_t *runouts) {
    int used[52] = {0};
    used[a[0]] = used[a[1]] = used[b[0]] = used[b[1]] = 1;
    for (int i = 0; i < nboard; i++) used[board[i]] = 1;
    int deck[52], nd = 0;
    for (int c = 0; c < 52; c++) if (!used[c]) deck[nd++] = c;

    int missing = 5 - nboard;
    int ha[7] = {a[0], a[1]}, hb[7] = {b[0], b[1]};
    for (int i = 0; i < nboard; i++) ha[2 + i] = hb[2 + i] = board[i];

    int idx[5];
    for (int i = 0; i < missing; i++) idx[i] = i;
    uint64_t wa = 0, wb = 0, n = 0;
    for (;;) {
        for (int i = 0; i < missing; i++) ha[2 + nboard + i] = hb[2 + nboard + i] = deck[idx[i]];
        uint32_t va = eval7(ha), vb = eval7(hb);
        if (va > vb) wa += 2;
        else if (vb > va) wb += 2;
        else { wa++; wb++; }
        n++;
        /* next combination of `missing` indices out of nd */
        int i = missing - 1;
        while (i >= 0 && idx[i] == nd - missing + i) i--;
        if (i < 0) break;
        idx[i]++;
        for (int j = i + 1; j < missing; j++) idx[j] = idx[j - 1] + 1;
    }
    *ua = wa; *ub = wb; *runouts = n;
}

/* Class index: 0..12 pairs 22..AA, then suited and offsuit by (high, low). */
static int class_of(int c1, int c2) {
    int r1 = card_rank(c1), r2 = card_rank(c2);
    if (r1 < r2) { int t = r1; r1 = r2; r2 = t; }
    if (r1 == r2) return r1;
    int base = 13 + (r1 * (r1 - 1)) / 2 + r2; /* 78 non-pair rank pairs */
    return card_suit(c1) == card_suit(c2) ? base : base + 78;
}

static void class_name(int cls, char *out) {
    if (cls < 13) { out[0] = out[1] = RANKS[cls]; out[2] = 0; return; }
    int idx = (cls - 13) % 78, suited = cls - 13 < 78;
    int r1 = 1;
    while ((r1 + 1) * r1 / 2 <= idx) r1++;
    int r2 = idx - r1 * (r1 - 1) / 2;
    out[0] = RANKS[r1]; out[1] = RANKS[r2]; out[2] = suited ? 's' : 'o'; out[3] = 0;
}

static int PERMS[24][4];

static void make_perms(void) {
    int n = 0;
    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) for (int c = 0; c < 4; c++) for (int d = 0; d < 4; d++)
        if (a != b && a != c && a != d && b != c && b != d && c != d) {
            PERMS[n][0] = a; PERMS[n][1] = b; PERMS[n][2] = c; PERMS[n][3] = d; n++;
        }
}

static uint32_t encode(int a0, int a1, int b0, int b1) {
    if (a0 < a1) { int t = a0; a0 = a1; a1 = t; }
    if (b0 < b1) { int t = b0; b0 = b1; b1 = t; }
    return (uint32_t)a0 << 24 | (uint32_t)a1 << 16 | (uint32_t)b0 << 8 | (uint32_t)b1;
}

/* Smallest encoding over suit relabellings and both orders; *swapped says whether
 * the first hand of the canonical form is b. */
static uint32_t canonical(const int *a, const int *b, int *swapped) {
    uint32_t best = UINT32_MAX;
    for (int p = 0; p < 24; p++) {
        int ma0 = card_rank(a[0]) * 4 + PERMS[p][card_suit(a[0])];
        int ma1 = card_rank(a[1]) * 4 + PERMS[p][card_suit(a[1])];
        int mb0 = card_rank(b[0]) * 4 + PERMS[p][card_suit(b[0])];
        int mb1 = card_rank(b[1]) * 4 + PERMS[p][card_suit(b[1])];
        uint32_t x = encode(ma0, ma1, mb0, mb1), y = encode(mb0, mb1, ma0, ma1);
        if (x < best) { best = x; *swapped = 0; }
        if (y < best) { best = y; *swapped = 1; }
    }
    return best;
}

static int cmp_u32(const void *x, const void *y) {
    uint32_t a = *(const uint32_t *)x, b = *(const uint32_t *)y;
    return a < b ? -1 : a > b;
}

typedef struct { int a[2], b[2]; uint32_t key; int swapped; } Pair;

static int run_table(const char *out_path) {
    make_perms();
    int hands[1326][2], nh = 0;
    for (int c1 = 0; c1 < 52; c1++) for (int c2 = 0; c2 < c1; c2++) { hands[nh][0] = c1; hands[nh][1] = c2; nh++; }

    size_t npairs = 0;
    Pair *pairs = malloc(sizeof(Pair) * 812175);
    for (int p = 0; p < nh; p++) for (int q = p + 1; q < nh; q++) {
        const int *a = hands[p], *b = hands[q];
        if (a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1]) continue;
        Pair *x = &pairs[npairs++];
        memcpy(x->a, a, sizeof x->a); memcpy(x->b, b, sizeof x->b);
        x->key = canonical(a, b, &x->swapped);
    }

    uint32_t *keys = malloc(sizeof(uint32_t) * npairs);
    for (size_t i = 0; i < npairs; i++) keys[i] = pairs[i].key;
    qsort(keys, npairs, sizeof(uint32_t), cmp_u32);
    size_t nkeys = 0;
    for (size_t i = 0; i < npairs; i++) if (i == 0 || keys[i] != keys[i - 1]) keys[nkeys++] = keys[i];
    fprintf(stderr, "%zu hand pairs, %zu distinct matchups up to suits\n", npairs, nkeys);

    uint64_t *units_first = malloc(sizeof(uint64_t) * nkeys), *units_second = malloc(sizeof(uint64_t) * nkeys);
    uint64_t boards = 0;
    int done = 0;
    #pragma omp parallel for schedule(dynamic, 16) reduction(max:boards)
    for (size_t i = 0; i < nkeys; i++) {
        uint32_t k = keys[i];
        int a[2] = {(int)(k >> 24), (int)(k >> 16 & 255)}, b[2] = {(int)(k >> 8 & 255), (int)(k & 255)};
        uint64_t n;
        matchup_units(a, b, NULL, 0, &units_first[i], &units_second[i], &n);
        if (n > boards) boards = n;
        #pragma omp atomic
        done++;
        if (done % 2000 == 0) fprintf(stderr, "  %d / %zu\n", done, nkeys);
    }

    static uint64_t count[NCLASS][NCLASS], units[NCLASS][NCLASS];
    for (size_t i = 0; i < npairs; i++) {
        Pair *x = &pairs[i];
        uint32_t *hit = bsearch(&x->key, keys, nkeys, sizeof(uint32_t), cmp_u32);
        size_t j = (size_t)(hit - keys);
        uint64_t ua = x->swapped ? units_second[j] : units_first[j];
        uint64_t ub = x->swapped ? units_first[j] : units_second[j];
        int ca = class_of(x->a[0], x->a[1]), cb = class_of(x->b[0], x->b[1]);
        count[ca][cb]++; units[ca][cb] += ua;
        count[cb][ca]++; units[cb][ca] += ub;
    }

    FILE *f = fopen(out_path, "w");
    if (!f) { perror(out_path); return 2; }
    fprintf(f, "# svlab-preflop-equity/1 boards=%llu units_per_board=2\n", (unsigned long long)boards);
    fprintf(f, "# class_i class_j pairs units_i\n");
    char ni[4], nj[4];
    for (int i = 0; i < NCLASS; i++) for (int j = 0; j < NCLASS; j++) {
        class_name(i, ni); class_name(j, nj);
        fprintf(f, "%s %s %llu %llu\n", ni, nj, (unsigned long long)count[i][j], (unsigned long long)units[i][j]);
    }
    fclose(f);
    return 0;
}

static int read_cards(const char *s, int *out, int max) {
    int n = 0;
    for (size_t i = 0; s[i]; i += 2) {
        if (n == max || !s[i + 1]) return -1;
        if ((out[n++] = parse_card(s + i)) < 0) return -1;
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "table") == 0) return run_table(argv[2]);
    if ((argc == 4 || argc == 5) && strcmp(argv[1], "matchup") == 0) {
        int a[2], b[2], board[5], nb = 0;
        if (read_cards(argv[2], a, 2) != 2 || read_cards(argv[3], b, 2) != 2) { fprintf(stderr, "bad hand\n"); return 2; }
        if (argc == 5 && (nb = read_cards(argv[4], board, 5)) < 0) { fprintf(stderr, "bad board\n"); return 2; }
        int used[52] = {0}, all[9] = {a[0], a[1], b[0], b[1]};
        for (int i = 0; i < nb; i++) all[4 + i] = board[i];
        for (int i = 0; i < 4 + nb; i++) { if (used[all[i]]++) { fprintf(stderr, "card repeated\n"); return 2; } }
        uint64_t ua, ub, n;
        matchup_units(a, b, board, nb, &ua, &ub, &n);
        printf("%llu %llu %llu\n", (unsigned long long)n, (unsigned long long)ua, (unsigned long long)ub);
        return 0;
    }
    fprintf(stderr, "usage: preflop_equity table OUT | preflop_equity matchup H1 H2 [BOARD]\n");
    return 2;
}
