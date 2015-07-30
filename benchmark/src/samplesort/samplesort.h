#pragma once

namespace SampleSort
{
  void sort_by_key(unsigned int *keys, unsigned int *keysEnd, unsigned int *values);
  void sort_by_key(unsigned long long *keys, unsigned long long *keysEnd, unsigned int *values);
  void sort(unsigned int *keys, unsigned int *keysEnd);
  void sort(unsigned long long *keys, unsigned long long *keysEnd);
}
