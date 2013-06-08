/* 
   A C-program for MT19937, with initialization improved 2002/2/10.
   Coded by Takuji Nishimura and Makoto Matsumoto.
   This is a faster version by taking Shawn Cokus's optimization,
   Matthe Bellew's simplification, Isaku Wada's real version.

   Before using, initialize the state by using init_genrand(seed) 
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.keio.ac.jp/matumoto/emt.html
   email: matumoto@math.keio.ac.jp
*/

/*
   C++ codes by Kohei Takeda (k-tak@letter.or.jp)
   Redistribution terms are the same as the original ones above.
*/

#ifndef ___MERSENNE_TWISTER_RNG___
#define ___MERSENNE_TWISTER_RNG___

#include <ctime>
#include <cstdlib>
#include <cassert>


struct Mt32Traits
{
      typedef unsigned int		UINTTYPE;
      typedef signed int		INTTYPE;
      static const int			INTTYPE_BITS	= 32;
      static const unsigned int		MAXDOUBLEVAL	= 4294967295U; //2^32-1
      static const size_t		NN		= 624;
      static const size_t		MM		= 397;
      static const unsigned int		INITVAL		= 1812433253U;
      static const unsigned int		ARRAYINITVAL_0	= 19650218U;
      static const unsigned int		ARRAYINITVAL_1	= 1664525U;
      static const unsigned int		ARRAYINITVAL_2	= 1566083941U;

      static unsigned int twist(const unsigned int& u, const unsigned int& v)
      {
	 static unsigned int mag01[2] = {0U, 0x9908b0dfU};
	 return ((((u & 0x80000000U) | (v & 0x7fffffffU)) >> 1) ^ mag01[v&1]);
      }
      
      static unsigned int temper(unsigned int y)
      {
	 y ^= (y >> 11);
	 y ^= (y << 7) & 0x9d2c5680U;
	 y ^= (y << 15) & 0xefc60000U;
	 y ^= (y >> 18);
	 
	 return y;
      }
};

struct Mt64Traits
{
      typedef unsigned long long	UINTTYPE;
      typedef signed long long		INTTYPE;
      static const int			INTTYPE_BITS	= 64;
      static const unsigned long long	MAXDOUBLEVAL	= 9007199254740991ULL; // 2^53-1
      static const size_t		NN		= 312;
      static const size_t		MM		= 156;
      static const unsigned long long	INITVAL		= 6364136223846793005ULL;
      static const unsigned long long	ARRAYINITVAL_0	= 19650218ULL;
      static const unsigned long long	ARRAYINITVAL_1	= 3935559000370003845ULL;
      static const unsigned long long	ARRAYINITVAL_2	= 2862933555777941757ULL;

      static unsigned long long twist(const unsigned long long& u, const unsigned long long& v)
      {
	 static unsigned long long mag01[2] = {0ULL, 0xB5026F5AA96619E9ULL};
	 return ((((u & 0xFFFFFFFF80000000ULL) | (v & 0x7FFFFFFFULL)) >> 1) ^ mag01[v&1]);
      }

      static unsigned long long temper(unsigned long long y)
      {
	 y ^= (y >> 29) & 0x5555555555555555ULL;
	 y ^= (y << 17) & 0x71D67FFFEDA60000ULL;
	 y ^= (y << 37) & 0xFFF7EEE000000000ULL;
	 y ^= (y >> 43);
	 
	 return y;
      }
};


template <typename Traits>
class MtRng
{
   public:
      typedef typename Traits::UINTTYPE	UINTTYPE;
      typedef typename Traits::INTTYPE	INTTYPE;

   protected:
      // member variables
      UINTTYPE*		state_;
      size_t		left_;
      UINTTYPE*		next_;
      
   protected:
      void nextState()
      {
	 UINTTYPE *p = state_;
	 size_t j;
	 
	 left_ = Traits::NN;
	 next_ = state_;
	 
	 for (j=Traits::NN-Traits::MM+1; --j; p++)
	    *p = p[Traits::MM] ^ Traits::twist(p[0], p[1]);
	 
	 for (j=Traits::MM; --j; p++)
	    *p = p[Traits::MM-Traits::NN] ^ Traits::twist(p[0], p[1]);
	 
	 *p = p[Traits::MM-Traits::NN] ^ Traits::twist(p[0], state_[0]);
      }
   
   public:
      MtRng()
      {
	 left_ = 1;
	 next_ = NULL;
	 state_ = (UINTTYPE*)malloc(sizeof(UINTTYPE) * Traits::NN);
	 init((UINTTYPE)time(NULL));
      }

      MtRng(UINTTYPE seed)
      {
	 left_ = 1;
	 next_ = NULL;
	 state_ = (UINTTYPE*)malloc(sizeof(UINTTYPE) * Traits::NN);
	 init(seed);
      }

      MtRng(UINTTYPE initkeys[], size_t keylen)
      {
	 left_ = 1;
	 next_ = NULL;
	 state_ = (UINTTYPE*)malloc(sizeof(UINTTYPE) * Traits::NN);
	 init(initkeys, keylen);
      }

      virtual ~MtRng()
      {
	 if (state_) {
	    free(state_);
	 }
      }
      
      void init(UINTTYPE seed)
      {
	 assert(sizeof(UINTTYPE)*8 == (size_t)Traits::INTTYPE_BITS);
	 
	 state_[0]= seed;
	 for (size_t j=1; j<Traits::NN; j++) {
	    state_[j]
	       = (Traits::INITVAL * (state_[j-1] ^ (state_[j-1] >> (Traits::INTTYPE_BITS-2)))
		  + (UINTTYPE)j); 
	 }
	 left_ = 1;
      }
      
      void init(UINTTYPE initkeys[], size_t keylen)
      {
	 init(Traits::ARRAYINITVAL_0);
	 
	 size_t i = 1;
	 size_t j = 0;
	 size_t k = (Traits::NN > keylen ? Traits::NN : keylen);
	 
	 for (; k; k--) {
	    state_[i]
	       = (state_[i]
		  ^ ((state_[i-1] ^ (state_[i-1] >> (Traits::INTTYPE_BITS-2)))
		     * Traits::ARRAYINITVAL_1))
	       + initkeys[j] + (UINTTYPE)j; /* non linear */
	    
	    i++;
	    j++;
	    
	    if (i >= Traits::NN) {
	       state_[0] = state_[Traits::NN-1];
	       i = 1;
	    }
	    if (j >= keylen) {
	       j = 0;
	    }
	 }
	 
	 for (k=Traits::NN-1; k; k--) {
	    state_[i]
	       = (state_[i]
		  ^ ((state_[i-1] ^ (state_[i-1] >> (Traits::INTTYPE_BITS-2)))
		     * Traits::ARRAYINITVAL_2))
	       - (UINTTYPE)i; /* non linear */
	    
	    i++;
	    
	    if (i >= Traits::NN) {
	       state_[0] = state_[Traits::NN-1];
	       i = 1;
	    }
	 }
	 
	 /* MSB is 1; assuring non-zero initial array */ 
	 state_[0] = (UINTTYPE)1 << (Traits::INTTYPE_BITS-1);
	 left_ = 1;
      }
      
      /* generates a random number on [0,2^bits-1]-interval */
      UINTTYPE getUint()
      {
	 if (--left_ == 0) nextState();
	 return Traits::temper(*next_++);
      }
      
      /* generates a random number on [0,2^(bits-1)-1]-interval */
      INTTYPE getInt()
      {
	 if (--left_ == 0) nextState();
	 return (INTTYPE)(Traits::temper(*next_++)>>1);
      }
      
      /* generates a random number on [0,1]-real-interval */
      double getReal1()
      {
	 if (--left_ == 0) nextState();
	 if (Traits::INTTYPE_BITS > 53) {
	    return (
	       (double)(Traits::temper(*next_++)>>(Traits::INTTYPE_BITS-53))
	       * (1.0 / 9007199254740991.0)
	       );
	 } else {
	    return (
	       (double)Traits::temper(*next_++) * (1.0/(double)Traits::MAXDOUBLEVAL)
	       );
	 }
      }
      
      /* generates a random number on [0,1)-real-interval */
      double getReal2()
      {
	 if (--left_ == 0) nextState();
	 if (Traits::INTTYPE_BITS > 53) {
	    return (
	       (double)(Traits::temper(*next_++)>>(Traits::INTTYPE_BITS-53))
	       * (1.0 / 9007199254740992.0)
	       );
	 } else {
	    return (
	       (double)Traits::temper(*next_++) * (1.0/((double)Traits::MAXDOUBLEVAL+1.0))
	       );
	 }
      }
      
      /* generates a random number on (0,1)-real-interval */
      double getReal3()
      {
	 if (--left_ == 0) nextState();
	 if (Traits::INTTYPE_BITS > 52) {
	    return (
	       ((double)(Traits::temper(*next_++)>>(Traits::INTTYPE_BITS-52)) + 0.5)
	       * (1.0 / 4503599627370496.0)
	       );
	 } else {
	    return (
	       ((double)Traits::temper(*next_++) + 0.5) * (1.0/((double)Traits::MAXDOUBLEVAL+1.0))
	       );
	 }
      }
};


typedef MtRng<Mt32Traits> MtRng32;
typedef MtRng<Mt64Traits> MtRng64;

#endif

