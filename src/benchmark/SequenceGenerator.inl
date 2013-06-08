#ifndef Sequence_Generator_inl
#define Sequence_Generator_inl

#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <climits>
#include <math.h>

namespace GpuSortingBenchmark
{
  namespace SequenceGenerator
  {
    const unsigned int DIST_COUNT = 9;

    template <typename element> 
    element randVal(MtRng32 *rng32, MtRng64 *rng64, unsigned int numBits, unsigned int numSamples, bool use64bits)
    {
      element val;
      if (!use64bits) val = (element)(rng32->getUint());
      else val = (element)(rng64->getUint());
      for(unsigned int i = 1; i < numSamples; i++)
      {
        if (!use64bits) val &= (element)(rng32->getUint());
        else val &= (element)(rng64->getUint());
      }

      if (!use64bits) return val >> (32 - numBits);  
      else return val >> (64 - numBits);  
    }

    template <typename element>
    void fill(MtRng32 *rng32, MtRng64 *rng64, element *data, unsigned int size, int distType, 
	unsigned int numBits, unsigned int numSamples, unsigned int P, unsigned int G, element RANGE, bool use64bits)
    {
      unsigned int b = numBits; unsigned int s = numSamples;
      if (!use64bits && numBits > 32)
      {
        std::cout << "Number of bits has to be in the range of [0, 32]";
        return;
      }
      else if (numBits > 64)
      {
        std::cout << "Number of bits has to be in the range of [0, 64]";
        return;
      }

      element MAX;
      if (!use64bits) MAX = (element)0xffffffff;
      else MAX = (element)0xffffffffffffffff;

      unsigned int c = 0;
      element val = 0;
      // Used for g-Group.
      element val3 = 0;
      element val2 = 0;
      element *T = new element[(int)RANGE];
      element tSum = 0;

      switch (distType)
      {
        // Constant.
      case 0:
        val = randVal<element>(rng32, rng64, b, s, use64bits);
        std::fill(data, data + size, randVal<element>(rng32, rng64, b, s, use64bits));
        break;
        // Uniform.
      case 1:
        for(unsigned int i = 0; i < size; i++) 
          data[i] = randVal<element>(rng32, rng64, b, s, use64bits);
        break;
        // Gaussian - average of 4.
      case 2:
        for(unsigned int i = 0; i < size; i++) 
          data[i] = (randVal<element>(rng32, rng64, b, s, use64bits) + randVal<element>(rng32, rng64, b, s, use64bits) + 
		randVal<element>(rng32, rng64, b, s, use64bits) + randVal<element>(rng32, rng64, b, s,use64bits)) / 4;
        break;
        // Bucket. This is different from the one used in the GPU Quicksort publication,
        // but like described here: http://www.umiacs.umd.edu/research/EXPAR/papers/3669/node5.html#SECTION00041000000000000000
      case 3:
        c = 0;
        val = (element)(MAX / P + 1);

        for (unsigned int i = 0; i < P; i++)
        {
          for (unsigned int j = 0; j < P; j++)
          {
            for (unsigned int k = 0; k < size / (P*P); k++)
            {
              // data[c] = (j * val) + randVal(b, s);
              data[c] = (j * val) + (randVal<element>(rng32, rng64, b, s, use64bits) % val);
              c++;
            }
          }
        }

        for (unsigned int i = c; i < size; i++)
          data[i] = randVal<element>(rng32, rng64, b, s, use64bits);

        break;
        // Staggered.
      case 4:
        c = 0;
        for (unsigned int i = 0; i < P; i++)
        {
          if (i < (P / 2))
            val = (2 * P) + 1;
          else
            val = (i - (P / 2)) * 2;

          val = val * ((MAX / P) + 1);

          for (unsigned int j = 0; j < size / P; j++)
          {
            data[c] = val + (randVal<element>(rng32, rng64, b, s, use64bits) / P) + 1;
            c++;
          }
        }

        for (unsigned int i = c; i < size; i++)
          data[i] = randVal<element>(rng32, rng64, b, s, use64bits);			

        break;
        // Sorted ascending.
      case 5:
        for(unsigned int i = 0; i < size; i++) 
          data[i] = randVal<element>(rng32, rng64, b, s, use64bits);
        std::sort(data, data + size);
        break;
        // g-Group.
      case 6: 
        c = 0;
        val = (MAX / P + 1);

        for (unsigned int i = 0; i < P; i++)
        {
          for (unsigned int j = 0; j < G; j++)
          {
            val2 = ((j * G) + (P / 2) + j) % P;
            val3 = ((j * G) + (P / 2) + j + 1) % P;
            for (unsigned int k = 0; k < size / (P*G); k++)
            {
              data[c] = val * val2 + randVal<element>(rng32, rng64, b, s, use64bits) % (val * (val3 - val2) - 1);
              c++;
            }
          }
        }

        for (unsigned int i = c; i < size; i++)
          data[i] = randVal<element>(rng32, rng64, b, s, use64bits);

        break;
        // Randomized duplicates.
      case 7: 
        c = 0;

        for (unsigned int i = 0; i < P; i++)
        {
          tSum = 0;
          for (unsigned int j = 0; j < RANGE; j++)
          {
            T[j] = randVal<element>(rng32, rng64, b, s, use64bits) % RANGE;
            tSum += T[j];
          }
          for (unsigned int j = 0; j < RANGE; j++)
          {
            val = randVal<element>(rng32, rng64, b, s, use64bits) % RANGE;
            for (unsigned int k = 0; k < (T[j] * size/P)/tSum ; k++)
            {
              data[c] = val;
              c++;
            }
          }
        }

        for (unsigned int i = c; i < size; i++)
          data[i] = randVal<element>(rng32, rng64, b, s, use64bits);

        break;
        // Deterministic duplicates.
      case 8:
        c = 0;
        for (unsigned int i = 2; i < P; i *=2)
        {
          for (unsigned int j = 0; j < P/i; j++)
          {
            for (unsigned int k = 0; k < size / P; k++)
            {
              data[c] = (element)(log((double)size * 2 / i) / log(2.0));
              c++;
            }
          }                                  
        }

        for (unsigned int i = 2; i < P; i *=2)
        {
          for (unsigned int j = 0; j < P/i; j++)
          {
            data[c] = (unsigned int)(log((double)size * 2 / i) / log(2.0));
            c++;
          }                                  
        }

        for (unsigned int i = c; i < size; i++)
          data[i] = randVal<element>(rng32, rng64, b, s, use64bits);
        break;
        // Sorted descending.
      case 9:
        for(unsigned int i = 0; i < size; i++) 
          data[i] = randVal<element>(rng32, rng64, b, s, use64bits);
        std::sort(data, data + size, std::greater<element>());
        break;
      default:
        std::cout << "Distribution not implemented.";
        break;
      }

      delete [] T;
    }
  }

}

#endif
