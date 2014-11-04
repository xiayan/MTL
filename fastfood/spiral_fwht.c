#include <mex.h>
#include <math.h>
#include "string.h"
#include "spiral_wht.h"

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[])
{
  double *input, *output;
  unsigned int rows, columns, i;
  Wht *wht_tree;

  if(nrhs!=1) mexErrMsgTxt("One input required.");

  input   = mxGetPr(prhs[0]);
  rows    = mxGetM (prhs[0]);
  columns = mxGetN (prhs[0]);

  if(rows&(rows-1))
    mexErrMsgTxt("Rows must be power of 2 greater than 1.");

  output = (double *) mxMalloc(rows*columns*sizeof(double));
  memcpy(output,input,rows*columns*sizeof(double));

  if(rows > 1) {
    wht_tree  = wht_get_tree((unsigned int) (log(rows)/log(2)));

    for(i = 0; i < columns; i++)
      wht_apply(wht_tree, 1, output+i*rows);

    wht_delete(wht_tree);
  }

  plhs[0] = mxCreateDoubleMatrix(rows,columns,mxREAL);
  mxSetPr(plhs[0], (double *) output);
}

