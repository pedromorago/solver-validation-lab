/* Shared by the C tools: cards as rank * 4 + suit (the same encoding as
 * svlab.cards), a seven-card evaluator that orders hands exactly like
 * svlab.evaluator.evaluate, and the 169 preflop hand classes in the order of
 * svlab.preflop.CLASSES. */
#ifndef SVLAB_POKER_H
#define SVLAB_POKER_H

#include <stdint.h>
#include <string.h>

#define NCLASS 169
static const char RANKS[] = "23456789TJQKA";
static const char SUITS[] = "cdhs";

static inline int card_rank(int c) { return c >> 2; }
static inline int card_suit(int c) { return c & 3; }

static inline int parse_card(const char *s) {
    const char *r = strchr(RANKS, s[0]);
    const char *u = strchr(SUITS, s[1]);
    if (!r || !u || !s[0] || !s[1]) return -1;
    return (int)(r - RANKS) * 4 + (int)(u - SUITS);
}

/* Highest card of the best straight in a 13-bit rank mask, or -1. The wheel is five-high (3). */
static inline int straight_high(unsigned mask) {
    for (int high = 12; high >= 4; high--) {
        unsigned need = 0x1Fu << (high - 4);
        if ((mask & need) == need) return high;
    }
    if ((mask & 0x100Fu) == 0x100Fu) return 3;
    return -1;
}

/* Packs a category and up to five ranks so that a bigger value is a better hand,
 * matching the tuples of svlab.evaluator.evaluate. */
static inline uint32_t pack(int cat, const int *k, int n) {
    uint32_t v = (uint32_t)cat << 20;
    for (int i = 0; i < n; i++) v |= (uint32_t)k[i] << (16 - 4 * i);
    return v;
}

static inline uint32_t eval7(const int *cards) {
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

/* Class index: 0..12 pairs 22..AA, then suited and offsuit by (high, low). */
static inline int class_of(int c1, int c2) {
    int r1 = card_rank(c1), r2 = card_rank(c2);
    if (r1 < r2) { int t = r1; r1 = r2; r2 = t; }
    if (r1 == r2) return r1;
    int base = 13 + (r1 * (r1 - 1)) / 2 + r2; /* 78 non-pair rank pairs */
    return card_suit(c1) == card_suit(c2) ? base : base + 78;
}

static inline void class_name(int cls, char *out) {
    if (cls < 13) { out[0] = out[1] = RANKS[cls]; out[2] = 0; return; }
    int idx = (cls - 13) % 78, suited = cls - 13 < 78;
    int r1 = 1;
    while ((r1 + 1) * r1 / 2 <= idx) r1++;
    int r2 = idx - r1 * (r1 - 1) / 2;
    out[0] = RANKS[r1]; out[1] = RANKS[r2]; out[2] = suited ? 's' : 'o'; out[3] = 0;
}

#endif
