/*
 * Three-handed preflop all-in equities for every triple of hand classes,
 * estimated by Monte Carlo, with exact card-removal weights.
 *
 *   threeway_equity table OUT [SAMPLES] [SEED]
 *   threeway_equity matchup H1 H2 H3 SAMPLES SEED
 *
 * For each triple a <= b <= c of the 169 classes (818,805 of them), in the
 * order a, then b, then c:
 *
 * - weight: the exact number of ways to deal one hand of class a, one of b
 *   and one of c with no card in common (the three seats are distinct);
 * - SAMPLES deals: a uniformly random such triple of hands and a random
 *   board from the 46 remaining cards. Each deal is worth 6 units in the
 *   three-way pot, split evenly among the hands that tie for best, and 2
 *   units in each of the three heads-up pots (a against b, a against c,
 *   b against c) with the third hand's cards dead.
 *
 * The output is binary, little-endian: a 16-byte header ("SVLAB3W1", then
 * SAMPLES and SEED as uint32) and seven uint16 per triple: weight, the
 * three-way units of a, b and c, and the heads-up units of a vs b, a vs c
 * and b vs c (for the first hand of each pair). Triples that can't be dealt
 * (AA AA AA) have weight 0 and no samples.
 *
 * The random numbers come from a counter-based generator keyed by SEED and
 * the triple's index, so the file is the same whatever the number of threads.
 *
 * Build: gcc -O3 -march=native -fopenmp -o threeway_equity threeway_equity.c
 */
#include <stdio.h>
#include <stdlib.h>

#include "poker.h"

static uint64_t splitmix(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static uint32_t below(uint64_t *state, uint32_t n) { return (uint32_t)((splitmix(state) >> 32) * n >> 32); }

static int COMBOS[NCLASS][12][2], NCOMBOS[NCLASS];

static void make_combos(void) {
    for (int c1 = 0; c1 < 52; c1++)
        for (int c2 = 0; c2 < c1; c2++) {
            int k = class_of(c1, c2);
            COMBOS[k][NCOMBOS[k]][0] = c1;
            COMBOS[k][NCOMBOS[k]][1] = c2;
            NCOMBOS[k]++;
        }
}

static int disjoint(const int *x, const int *y) { return x[0] != y[0] && x[0] != y[1] && x[1] != y[0] && x[1] != y[1]; }

typedef struct { uint16_t weight, three[3], two[3]; } Row;

static void run_triple(int a, int b, int c, uint32_t samples, uint64_t seed, uint64_t index, Row *row) {
    static __thread int list[1728][3];
    int n = 0;
    for (int x = 0; x < NCOMBOS[a]; x++)
        for (int y = 0; y < NCOMBOS[b]; y++) {
            if (!disjoint(COMBOS[a][x], COMBOS[b][y])) continue;
            for (int z = 0; z < NCOMBOS[c]; z++)
                if (disjoint(COMBOS[a][x], COMBOS[c][z]) && disjoint(COMBOS[b][y], COMBOS[c][z])) {
                    list[n][0] = x; list[n][1] = y; list[n][2] = z; n++;
                }
        }
    memset(row, 0, sizeof *row);
    row->weight = (uint16_t)n;
    if (n == 0) return;

    uint64_t state = seed * 0xD1B54A32D192ED03ull ^ index;
    uint32_t three[3] = {0}, two[3] = {0};
    for (uint32_t s = 0; s < samples; s++) {
        int *pick = list[below(&state, (uint32_t)n)];
        const int *h[3] = {COMBOS[a][pick[0]], COMBOS[b][pick[1]], COMBOS[c][pick[2]]};
        int used[52] = {0}, deck[46], nd = 0;
        for (int p = 0; p < 3; p++) used[h[p][0]] = used[h[p][1]] = 1;
        for (int card = 0; card < 52; card++) if (!used[card]) deck[nd++] = card;
        for (int i = 0; i < 5; i++) { /* partial Fisher-Yates: the board is deck[0..4] */
            int j = i + (int)below(&state, (uint32_t)(nd - i));
            int t = deck[i]; deck[i] = deck[j]; deck[j] = t;
        }
        uint32_t v[3];
        for (int p = 0; p < 3; p++) {
            int cards[7] = {h[p][0], h[p][1], deck[0], deck[1], deck[2], deck[3], deck[4]};
            v[p] = eval7(cards);
        }
        uint32_t best = v[0] > v[1] ? (v[0] > v[2] ? v[0] : v[2]) : (v[1] > v[2] ? v[1] : v[2]);
        int winners = (v[0] == best) + (v[1] == best) + (v[2] == best);
        for (int p = 0; p < 3; p++) if (v[p] == best) three[p] += 6 / winners;
        static const int PAIRS[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        for (int q = 0; q < 3; q++) {
            uint32_t x = v[PAIRS[q][0]], y = v[PAIRS[q][1]];
            two[q] += x > y ? 2 : x == y ? 1 : 0;
        }
    }
    for (int p = 0; p < 3; p++) { row->three[p] = (uint16_t)three[p]; row->two[p] = (uint16_t)two[p]; }
}

static int class_index(const char *name) {
    char buf[4];
    for (int k = 0; k < NCLASS; k++) { class_name(k, buf); if (strcmp(buf, name) == 0) return k; }
    return -1;
}

int main(int argc, char **argv) {
    make_combos();
    if (argc >= 3 && argc <= 5 && strcmp(argv[1], "table") == 0) {
        uint32_t samples = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4000;
        uint64_t seed = argc > 4 ? strtoull(argv[4], NULL, 10) : 1;
        if (samples == 0 || samples * 6ull > 65535) { fprintf(stderr, "SAMPLES must be 1 to 10922\n"); return 2; }
        size_t total = 0;
        for (int a = 0; a < NCLASS; a++) for (int b = a; b < NCLASS; b++) total += (size_t)(NCLASS - b);
        Row *rows = calloc(total, sizeof(Row));
        size_t *starts = malloc(sizeof(size_t) * NCLASS * NCLASS);
        size_t at = 0;
        for (int a = 0; a < NCLASS; a++) for (int b = a; b < NCLASS; b++) { starts[a * NCLASS + b] = at; at += (size_t)(NCLASS - b); }

        #pragma omp parallel for schedule(dynamic, 1)
        for (int ab = 0; ab < NCLASS * NCLASS; ab++) {
            int a = ab / NCLASS, b = ab % NCLASS;
            if (b < a) continue;
            for (int c = b; c < NCLASS; c++) {
                size_t index = starts[ab] + (size_t)(c - b);
                run_triple(a, b, c, samples, seed, index, &rows[index]);
            }
        }

        FILE *f = fopen(argv[2], "wb");
        if (!f) { perror(argv[2]); return 2; }
        uint32_t header[2] = {samples, (uint32_t)seed};
        fwrite("SVLAB3W1", 1, 8, f);
        fwrite(header, sizeof header, 1, f);
        for (size_t i = 0; i < total; i++) {
            uint16_t out[7] = {rows[i].weight, rows[i].three[0], rows[i].three[1], rows[i].three[2],
                               rows[i].two[0], rows[i].two[1], rows[i].two[2]};
            fwrite(out, sizeof out, 1, f);
        }
        fclose(f);
        fprintf(stderr, "%zu triples, %u samples each\n", total, samples);
        return 0;
    }
    if (argc == 7 && strcmp(argv[1], "matchup") == 0) {
        int k[3];
        for (int p = 0; p < 3; p++) if ((k[p] = class_index(argv[2 + p])) < 0) { fprintf(stderr, "bad class %s\n", argv[2 + p]); return 2; }
        Row row;
        run_triple(k[0], k[1], k[2], (uint32_t)strtoul(argv[5], NULL, 10), strtoull(argv[6], NULL, 10), 0, &row);
        printf("%u %u %u %u %u %u %u\n", row.weight, row.three[0], row.three[1], row.three[2], row.two[0], row.two[1], row.two[2]);
        return 0;
    }
    fprintf(stderr, "usage: threeway_equity table OUT [SAMPLES] [SEED] | threeway_equity matchup C1 C2 C3 SAMPLES SEED\n");
    return 2;
}
