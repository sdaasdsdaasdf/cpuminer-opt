#include "algo-gate-api.h"
#include "algo/sha/sha256-hash.h"
#include "Verthash.h"
#include "tiny_sha3/sha3-4way.h"

/*
 * Global Verthash info
 * Loaded once, reused by all threads
 * Contains pointer to verthash.dat and its size
 */
static verthash_info_t verthashInfo;

/*
 * Expected SHA-256 hash of verthash.dat
 * Used only for verification at startup
 * NEVER used during mining
 */
static const uint8_t verthashDatFileHash_bytes[32] =
{
  0xa5, 0x55, 0x31, 0xe8, 0x43, 0xcd, 0x56, 0xb0,
  0x10, 0x11, 0x4a, 0xaf, 0x63, 0x25, 0xb0, 0xd5,
  0x29, 0xec, 0xf8, 0x8f, 0x8a, 0xd4, 0x76, 0x39,
  0xb6, 0xed, 0xed, 0xaf, 0xd7, 0x21, 0xaa, 0x48
};

/*
 * Thread-local SHA3 midstates
 * Each mining thread owns its own copy
 * This avoids locks and cache contention
 */
#if defined(__AVX2__)
static __thread sha3_4way_ctx_t sha3_mid_ctxA;
static __thread sha3_4way_ctx_t sha3_mid_ctxB;
#else
static __thread sha3_ctx_t sha3_mid_ctx[8];
#endif

/*
 * Prehash first 72 bytes of block header using SHA3-512
 * This part of the header NEVER changes
 * Only nonce is added later
 */
void verthash_sha3_512_prehash_72(const void *restrict input)
{
#if defined(__AVX2__)

   __m256i vin[10];
   mm256_intrlv80_4x64(vin, input);

   sha3_4way_init(&sha3_mid_ctxA, 64);
   sha3_4way_init(&sha3_mid_ctxB, 64);

   /*
    * Prepare two SIMD lanes so we can hash
    * multiple nonces in parallel later
    */
   vin[0] = _mm256_add_epi8(vin[0], _mm256_set_epi64x(4, 3, 2, 1));
   sha3_4way_update(&sha3_mid_ctxA, vin, 72);

   vin[0] = _mm256_add_epi8(vin[0], _mm256_set1_epi64x(4));
   sha3_4way_update(&sha3_mid_ctxB, vin, 72);

#else

   /*
    * Fallback path: prepare 8 independent SHA3 contexts
    * Still efficient, just no SIMD
    */
   uint8_t in[80] __attribute__((aligned(64)));
   memcpy(in, input, 80);

   for (int i = 0; i < 8; i++)
   {
      in[0] += 1; // Slight nonce variation per lane
      sha3_init(&sha3_mid_ctx[i], 64);
      sha3_update(&sha3_mid_ctx[i], in, 72);
   }

#endif
}

/*
 * Finalize SHA3-512 hash for 8 nonces
 * Uses the precomputed midstate
 */
void verthash_sha3_512_final_8(void *restrict hash, uint64_t nonce)
{
#if defined(__AVX2__)

   __m256i vhashA[10] __attribute__((aligned(64)));
   __m256i vhashB[10] __attribute__((aligned(64)));
   sha3_4way_ctx_t ctx;

   const __m256i vnonce = _mm256_set1_epi64x(nonce);

   memcpy(&ctx, &sha3_mid_ctxA, sizeof(ctx));
   sha3_4way_update(&ctx, &vnonce, 8);
   sha3_4way_final(vhashA, &ctx);

   memcpy(&ctx, &sha3_mid_ctxB, sizeof(ctx));
   sha3_4way_update(&ctx, &vnonce, 8);
   sha3_4way_final(vhashB, &ctx);

   /*
    * De-interleave SIMD output into linear hashes
    */
   dintrlv_4x64(hash,       hash + 64,  hash + 128, hash + 192, vhashA, 512);
   dintrlv_4x64(hash + 256, hash + 320, hash + 384, hash + 448, vhashB, 512);

#else

   for (int i = 0; i < 8; i++)
   {
      sha3_ctx_t ctx;
      memcpy(&ctx, &sha3_mid_ctx[i], sizeof(ctx));
      sha3_update(&ctx, &nonce, 8);
      sha3_final((uint8_t *)hash + i * 64, &ctx);
   }

#endif
}

/*
 * MAIN MINING LOOP
 * This is pure brute force:
 * increment nonce → hash → compare → repeat
 */
int scanhash_verthash(struct work *restrict work,
                      uint32_t max_nonce,
                      uint64_t *restrict hashes_done,
                      struct thr_info *restrict mythr)
{
   uint32_t hash[8] __attribute__((aligned(64)));
   uint32_t edata[20] __attribute__((aligned(32)));

   uint32_t *pdata        = work->data;
   const uint32_t *ptarget = work->target;

   const uint32_t first_nonce = pdata[19];
   const uint32_t last_nonce  = max_nonce - 1;

   uint32_t n = first_nonce;
   const int thr_id = mythr->id;
   const bool bench = opt_benchmark;

   /*
    * Convert block header to big-endian once
    * Saves work inside the loop
    */
   v128_bswap32_80(edata, pdata);

   /*
    * Prehash static header part
    */
   verthash_sha3_512_prehash_72(edata);

   while (n < last_nonce && !work_restart[thr_id].restart)
   {
      edata[19] = n;

      /*
       * Heavy memory-hard hash
       * This dominates runtime
       */
      verthash_hash(verthashInfo.data,
                    verthashInfo.dataSize,
                    edata,
                    hash);

      /*
       * Very cheap comparison
       * Only rarely true
       */
      if (!bench && valid_hash(hash, ptarget))
      {
         pdata[19] = bswap_32(n);
         submit_solution(work, hash, mythr);
      }

      n++;
   }

   *hashes_done = n - first_nonce;
   pdata[19] = n;
   return 0;
}
