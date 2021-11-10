#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_int exp;
  __pp_vec_int zeros = _pp_vset_int(0);
  __pp_vec_int ones = _pp_vset_int(1);
  __pp_vec_float maxs = _pp_vset_float(EXP_MAX);
  __pp_mask mask_data, mask_op;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    if(i + VECTOR_WIDTH < N){
      mask_data = _pp_init_ones();
      mask_op = _pp_init_ones();
    }
    else{
      mask_data = _pp_init_ones(N - i);
      mask_op = _pp_init_ones(N - i);
    }

    _pp_vload_float(x, values + i, mask_data);//load value
    _pp_vload_int(exp, exponents + i, mask_data);//load exponents
    _pp_vset_float(result, 1.0, mask_data);//set result to 1's

    _pp_vgt_int(mask_op, exp, zeros, mask_data);
    while(_pp_cntbits(mask_op) > 0){
      _pp_vmult_float(result, result, x, mask_op);// result = result * x

      _pp_vsub_int(exp, exp, ones, mask_op);//exp--
      _pp_vgt_int(mask_op, exp, zeros, mask_data);//op = (exp > 0)

      __pp_mask exceed;
      _pp_vgt_float(exceed, result, maxs, mask_data);//if result > 9.99999
      _pp_vset_float(result, EXP_MAX, exceed); //then result = EXP_MAX
      exceed = _pp_mask_not(exceed);
      mask_op = _pp_mask_and(mask_op, exceed);

    }
    _pp_vstore_float(output + i, result, mask_data);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float result = 0;
  float dump[VECTOR_WIDTH];
  __pp_vec_float x;
  __pp_mask maskAll = _pp_init_ones();

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    int n = VECTOR_WIDTH;
    _pp_vload_float(x, values + i, maskAll);
    while(n > 1){
      _pp_hadd_float(x, x);
      _pp_interleave_float(x, x);
      n /= 2;
    }
    _pp_vstore_float(dump, x, maskAll);
    result += dump[0];
  }

  return result;
}