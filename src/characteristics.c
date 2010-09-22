#include "presto.h"

static double extended_equiv_gaussian_sigma(double logp);
static double log_asymtotic_incomplete_gamma(double a, double z);
static double log_asymtotic_gamma(double z);

float get_numphotons(FILE * file)
  /* Return the total number of photons in the FFT file      */
  /* i.e.  it returns the value of the 0th frequency bin.    */
  /* Arguments:                                              */
  /*   'file' is a pointer to the file you want to access.   */
{
   float nph;

   chkfileseek(file, 0, sizeof(fcomplex), SEEK_SET);
   chkfread(&nph, sizeof(float), 1, file);

   /* The following protects against pre-normalized time-series */

   if (nph <= 0)
      nph = 1.0;
   return nph;
}


double get_localpower(fcomplex * data, int numdata, double r)
  /* Return the local power level at specific FFT frequency.  */
  /* Arguments:                                               */
  /*   'data' is a pointer to a complex FFT.                  */
  /*   'numdata' is the number of complex points in 'data'.   */
  /*   'r' is the Fourier frequency in data that we want to   */
  /*      interpolate.                                        */
{
   double powargr, powargi, sum = 0.0;
   int ii, binsperside, lo1, lo2, hi1, hi2, intfreq;

   intfreq = (long) floor(r);
   binsperside = NUMLOCPOWAVG / 2;

   /* Set the bounds of our summation */

   lo1 = intfreq - DELTAAVGBINS - binsperside + 1;
   hi1 = lo1 + binsperside;
   lo2 = intfreq + DELTAAVGBINS + 1;
   hi2 = lo2 + binsperside;

   /* Make sure we don't try to read non-existant data */

   if (lo1 < 0)
      lo1 = 0;
   if (lo2 < 0)
      lo2 = 0;
   if (hi1 < 0)
      hi1 = 0;
   if (hi2 < 0)
      hi2 = 0;
   if (lo1 > numdata)
      lo1 = numdata;
   if (lo2 > numdata)
      lo2 = numdata;
   if (hi1 > numdata)
      hi1 = numdata;
   if (hi2 > numdata)
      hi2 = numdata;

   /* Perform the summation */

   for (ii = lo1; ii < hi1; ii++)
      sum += POWER(data[ii].r, data[ii].i);
   for (ii = lo2; ii < hi2; ii++)
      sum += POWER(data[ii].r, data[ii].i);
   sum /= (double) NUMLOCPOWAVG;
   return sum;
}


double get_localpower3d(fcomplex * data, int numdata, double r, double z, double w)
  /* Return the local power level around a specific FFT           */
  /* frequency, f-dot, and f-dotdot.                              */
  /* Arguments:                                                   */
  /*   'data' is a pointer to a complex FFT.                      */
  /*   'numdata' is the number of complex points in 'data'.       */
  /*   'r' is the Fourier frequency in data that we want to       */
  /*      interpolate.                                            */
  /*   'z' is the Fourier Frequency derivative (# of bins the     */
  /*       signal smears over during the observation).            */
  /*   'w' is the Fourier Frequency 2nd derivative (change in the */
  /*       Fourier f-dot during the observation).                 */
{
   double powargr, powargi, sum = 0.0;
   double lo1, lo2, hi1, hi2, freq;
   int binsperside, kern_half_width;
   fcomplex ans;

   binsperside = NUMLOCPOWAVG / 2;
   kern_half_width = w_resp_halfwidth(z, w, LOWACC);

   /* Set the bounds of our summation */

   lo1 = r - DELTAAVGBINS - binsperside;
   hi1 = lo1 + binsperside;
   lo2 = r + DELTAAVGBINS;
   hi2 = lo2 + binsperside;

   /* Make sure we don't try to read non-existant data */

   if (lo1 < 0.0)
      lo1 = 0.0;
   if (lo2 < 0.0)
      lo2 = 0.0;
   if (hi1 < 0.0)
      hi1 = 0.0;
   if (hi2 < 0.0)
      hi2 = 0.0;
   if (lo1 > numdata)
      lo1 = (double) numdata;
   if (lo2 > numdata)
      lo2 = (double) numdata;
   if (hi1 > numdata)
      hi1 = (double) numdata;
   if (hi2 > numdata)
      hi2 = (double) numdata;

   /* Perform the summation */

   for (freq = lo1; freq < hi1; freq += 1.0) {
      rzw_interp(data, numdata, freq, z, w, kern_half_width, &ans);
      sum += POWER(ans.r, ans.i);
   }
   for (freq = lo2; freq < hi2; freq += 1.0) {
      rzw_interp(data, numdata, freq, z, w, kern_half_width, &ans);
      sum += POWER(ans.r, ans.i);
   }
   sum /= (double) NUMLOCPOWAVG;
   return sum;
}


void get_derivs3d(fcomplex * data, int numdata, double r,
                  double z, double w, double localpower, rderivs * result)
  /* Return an rderives structure that contains the power,      */
  /* phase, and their first and second derivatives at a point   */
  /* in the F/F-dot/F-dortdot volume.                           */
  /* Arguments:                                                 */
  /*   'data' is a pointer to a complex FFT.                    */
  /*   'numdata' is the number of complex points in 'data'.     */
  /*   'r' is the Fourier frequency in data that we want to     */
  /*      interpolate.                                          */
  /*   'z' is the Fourier Frequency derivative (# of bins the   */
  /*       signal smears over during the observation).          */
  /*   'w' is the Fourier Frequency 2nd derivative (change in   */
  /*       the Fourier f-dot during the observation).           */
  /*   'localpower' is the local power level around the signal. */
  /*   'result' is a pointer to an rderivs structure that will  */
  /*       contain the results.                                 */
{
   /* h = Length of delta for derivatives (See Num Recip p. 186)  */
   /* This is optimized for single precision powers and phases.   */

   double h = 0.003, twoh = 0.006, twohsqrd = 0.000036, f;
   double powargr, powargi, radargr, radargi, radtmp, pwr[5], phs[5];
   int ii, kern_half_width;
   fcomplex ans;

   /* Read the powers and phases: */

   kern_half_width = w_resp_halfwidth(z, w, HIGHACC);
   for (ii = 0, f = r - twoh; ii < 5; ii++, f += h) {
      rzw_interp(data, numdata, f, z, w, kern_half_width, &ans);
      pwr[ii] = POWER(ans.r, ans.i);
      phs[ii] = RADIAN_PHASE(ans.r, ans.i);
   }

   /* Ensure there are no discontinuities in the phase values: */

   for (ii = 0; ii < 4; ii++) {
      if (fabs(phs[ii + 1] - phs[ii]) > PI)
         phs[ii + 1] -= TWOPI;
   }

   /* Calculate the derivatives */

   result->pow = pwr[2];
   result->phs = phs[2];
   result->dpow = (pwr[3] - pwr[1]) / twoh;
   result->dphs = (phs[3] - phs[1]) / twoh;
   result->d2pow = (pwr[4] - 2.0 * pwr[2] + pwr[0]) / twohsqrd;
   result->d2phs = (phs[4] - 2.0 * phs[2] + phs[0]) / twohsqrd;
   result->locpow = localpower;
}


void calc_props(rderivs data, double r, double z, double w, fourierprops * result)
  /* Return a fourierprops structure that contains the various  */
  /* properties of a signal described by Middleditch, Deich,    */
  /* and Kulkarni in _Isolated_Pulsars_, 1993, p372.            */
  /* Arguments:                                                 */
  /*   'data' is a pointer to an rderivs structure containing   */
  /*       derivative information about the peak in question.   */
  /*   'r' is the Fourier frequency in data that we want to     */
  /*      interpolate.                                          */
  /*   'z' is the Fourier Frequency derivative (# of bins the   */
  /*       signal smears over during the observation).          */
  /*   'w' is the Fourier Frequency second derivative.          */
  /*   'result' is a pointer to an fourierprops structure that  */
  /*       will contain the results.                            */
{
   double tmppow, tmpd2pow;

   /* Protect against division-by-zero */
   if (fabs(data.pow) < DBLCORRECT) {
      fprintf(stderr, "\n data.pow = %f (out-of-bounds) in calc_props().\n\n", data.pow);
      exit(-1);
   }
   if (data.locpow < DBLCORRECT) {
      fprintf(stderr, "\n data.locpow = %f (out-of-bounds) in calc_props().\n\n",
             data.locpow);
      exit(-1);
   }
   /* Correct the power and its derivatives to the local power level */
   tmppow = data.pow / data.locpow;
   tmpd2pow = data.d2pow / data.locpow;
   /* Purity */
   result->pur = sqrt(1.5 * fabs(tmpd2pow) / tmppow) / PI;
   /* The following kludge 'fixes' a floating point exception */
   /* that is found (very rarely) on DEC Alphas               */
   if (fabs(tmppow) < DBLCORRECT)
      result->pur = 1.0;
   /* Purity error */
   result->purerr = 1.0 / (result->pur * sqrt(10.0 * tmppow));
   /* Fourier frequency */
   result->r = r;
   /* Fourier frequency error */
   result->rerr = 3.0 / (PI * result->pur * sqrt(6.0 * tmppow));
   /* Fourier frequency derivative (f-dot) */
   result->z = z;
   /* Fourier frequency derivative error */
   result->zerr = 3.0 * sqrt(10.0) / (PI * result->pur * result->pur * sqrt(tmppow));
   /* Fourier frequency second derivative (f-dotdot) */
   result->w = w;
   /* Fourier frequency second derivative error */
   result->werr = 6.0 * sqrt(105.0) /
       (PI * result->pur * result->pur * result->pur * sqrt(tmppow));
   /* Normalized power */
   result->pow = tmppow;
   /* Normalized power error */
   result->powerr = sqrt(2.0 * tmppow);
   /* Signal significance in sigma */
   result->sig = sqrt(2.0 * tmppow - log(PI * tmppow));
   /* Raw power */
   result->rawpow = data.pow;
   /* Phase (radians) */
   result->phs = data.phs;
   /* Phase error (radians) */
   result->phserr = 1.0 / result->powerr;
   /* Centroid */
   result->cen = -data.dphs / TWOPI;
   /* Centroid error */
   result->cenerr = 1.0 / sqrt(24.0 * tmppow);
   /* Local power level */
   result->locpow = data.locpow;
}


void calc_binprops(fourierprops * props, double T, int lowbin,
                   int nfftbins, binaryprops * result)
  /* Return a binaryprops structure that contains the various     */
  /* estimates of the binary pulsar system from a mini-FFT.       */
  /* Arguments:                                                   */
  /*   'props' is a pointer to the candidate's fourierprops.      */
  /*   'T' is the total length (sec) of the original time series. */
  /*   'lowbin' is the Fourier bin number from the original FFT   */
  /*      the lowest bin in the mini-FFT.                         */
  /*   'nfftbins' is the number of bins in the mini-FFT.          */
  /*   'absnorm' is the value of the power normalization          */
  /*      constant for this mini-FFT.                             */
  /*   'result' is the returned binaryprops structure.            */
{
   if (T <= 0.0) {
      fprintf(stderr, "\n T = %f (out-of-bounds) in calc_binprops().\n\n", T);
      exit(-1);
   }
   if (nfftbins <= 0) {
      fprintf(stderr, "\n nfftbins = %d (out-of-bounds) in calc_binprops().\n\n", nfftbins);
      exit(-1);
   }
   /* Mini-FFT Raw power */
   result->pow = props->rawpow;
   /* Mini-FFT Raw power error */
   result->powerr = sqrt(2.0 * props->rawpow);
   /* # of bins in the Mini-FFT */
   result->nfftbins = nfftbins;
   /* Fourier bin number of the lowest freq bin Mini-FFT'd  */
   result->lowbin = lowbin;
   /* The following two are big assumptions... */
   /* Pulsar Fourier frequency estimate */
   result->rpsr = lowbin + props->cen * nfftbins;
   /* Pulsar Fourier frequency estimate error */
   result->rpsrerr = 2.0 * props->cenerr * nfftbins;
   /* Orbital frequency in bins */
   result->rbin = nfftbins / props->r;
   /* Orbital frequency error in bins */
   result->rbinerr = props->rerr * nfftbins / (props->r * props->r);
   /* Pulsar frequency estimate (hz) */
   result->fpsr = result->rpsr / T;
   /* Pulsar frequency estimate error (hz) */
   result->fpsrerr = result->rpsrerr / T;
   /* Pulsar period estimate (s) */
   result->ppsr = T / result->rpsr;
   /* Pulsar period estimate error (s) */
   result->ppsrerr = T * result->rpsrerr / (result->rpsr * result->rpsr);
   /* Orbital period (sec) */
   result->pbin = T / result->rbin;
   /* Orbital period error (sec) */
   result->pbinerr = T * props->rerr / nfftbins;
   /* The following four are big handwaves... */
   /* Phase modulation amplitude (radians) */
   result->z = 0.4 * nfftbins * props->pur / result->rbin;
   /* Phase modulation amplitude error (radians) */
   result->zerr = 2.0 * result->z *
       sqrt(pow(props->purerr / props->pur, 2.0) +
            pow(result->rbinerr / result->rbin, 2.0));
   /* Orbital semi-major axis estimate (lt-sec) */
   result->asinic = result->z / (result->fpsr * TWOPI);
   /* Orbital semi-major axis error estimate (lt-sec) */
   result->asinicerr = result->asinic *
       sqrt(pow(result->zerr / result->z, 2.0) +
            pow(result->fpsrerr / result->fpsr, 2.0));
   /* Mini-FFT bin where signal was detected */
   result->rdetect = props->r;
   /* Error in Mini-FFT bin where signal was detected */
   result->rdetecterr = props->rerr;
   /* Sigma of the Mini-FFT detection */
   result->sig = props->sig;
   /* Phase of the Mini-FFT detection (radians) */
   result->phs = props->phs;
   /* Phase error of the Mini-FFT signal */
   result->phserr = props->phserr;
   /* Purity of the Mini-FFT signal */
   result->pur = props->pur;
   /* Purity error of the Mini-FFT signal */
   result->purerr = props->purerr;
   /* Centroid of the Mini-FFT signal */
   result->cen = props->cen;
   /* Centroid error of the Mini-FFT signal */
   result->cenerr = props->cenerr;
}


void calc_rzwerrs(fourierprops * props, double T, rzwerrs * result)
  /* Calculate periods, frequencies, their derivatives        */
  /* and their errors.                                        */
  /* Arguments:                                               */
  /*   'props' is a pointer to a fourierprops structure.      */
  /*   'T' is the length of the data set in sec (i.e. N*dt).  */
  /*   'result' is a pointer to the returned rzwerrs struct.  */
{
   double tmp, T2, T3, r2, r4, z2, sr2, sz2;

   if (T <= 0.0) {
      fprintf(stderr, "\n T = %f (out-of-bounds) in calc_rzwerrs().\n\n", T);
      exit(-1);
   }
   /* prep some useful values */

   T2 = T * T;
   T3 = T2 * T;
   r2 = props->r * props->r;
   sr2 = props->rerr * props->rerr;
   r4 = r2 * r2;
   z2 = props->z * props->z;
   sz2 = props->zerr * props->zerr;
   tmp = props->r * props->w - 3 * z2;

   /* Do the calculations */

   result->f = props->r / T;
   result->ferr = props->rerr / T;
   result->fd = props->z / T2;
   result->fderr = props->zerr / T2;
   result->fdd = props->w / T3;
   result->fdderr = props->werr / T3;
   result->p = T / props->r;
   result->perr = T * props->rerr / r2;
   result->pd = -props->z / r2;
   result->pderr = sqrt(4 * z2 * sr2 / (r4 * r2) + sz2 / r4);
   if (props->w == 0.0)
      result->pdd = 0.0;
   else
      result->pdd = (2 * z2 - props->r * props->w) / (T * r2 * props->r);
   result->pdderr = sqrt((props->werr * props->werr * r4 + 16 * sz2 * r2 +
                          4 * tmp * tmp * sr2) / (r4 * r4 * T2));
}


double extended_equiv_gaussian_sigma(double logp)
/*
  extended_equiv_gaussian_sigma(double logp):
      Return the equivalent gaussian sigma corresponding
          to the log of the cumulative gaussian probability logp.
          In other words, return x, such that Q(x) = p, where Q(x)
          is the cumulative normal distribution.  This version uses
          the rational approximation from Abramowitz and Stegun,
          eqn 26.2.23.  Using the log(P) as input gives a much
          extended range.
*/
{
   double t, num, denom;

   t = sqrt(-2.0 * logp);
   num = 2.515517 + t * (0.802853 + t * 0.010328);
   denom = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308));
   return t - num / denom;
}


double log_asymtotic_incomplete_gamma(double a, double z)
/*
  log_asymtotic_incomplete_gamma(double a, double z):
      Return the log of the incomplete gamma function in its
          asymtotic limit as z->infty.  This is from Abramowitz
          and Stegun eqn 6.5.32.
*/
{
   double x = 1.0, newxpart = 1.0, term = 1.0;
   int ii = 1;

   while (fabs(newxpart) > 1e-15) {
      term *= (a - ii);
      newxpart = term / pow(z, ii);
      x += newxpart;
      ii += 1;
   }
   return (a - 1.0) * log(z) - z + log(x);
}

double log_asymtotic_gamma(double z)
/*
  log_asymtotic_gamma(double z):
      Return the log of the gamma function in its asymtotic limit
          as z->infty.  This is from Abramowitz and Stegun eqn 6.1.41.
*/
{
   double x, y;

   x = (z - 0.5) * log(z) - z + 0.91893853320467267;
   y = 1.0 / (z * z);
   x += (((-5.9523809523809529e-4 * y
           + 7.9365079365079365079365e-4) * y
          - 2.7777777777777777777778e-3) * y + 8.3333333333333333333333e-2) / z;
   return x;
}

double candidate_sigma(double power, int numsum, double numtrials)
/* Return the approximate significance in Gaussian       */
/* sigmas of a candidate of numsum summed powers,        */
/* taking into account the number of independent trials. */
{
   double x = 0.0;

   if (power <= 0.0) {
      return 0.0;
   }

   if (power > 100.0) {
      double logp;

      /* Use some asymtotic expansions for the chi^2 distribution */
      logp = log_asymtotic_incomplete_gamma(numsum, power) -
          log_asymtotic_gamma(numsum);
      /* Now adjust for the number of trials */
      logp += log(numtrials);
      /* Convert to a sigma */
      x = extended_equiv_gaussian_sigma(logp);
   } else {
      int which, status;
      double p, q, bound, mean = 0.0, sd = 1.0, shape, scale = 1.0;

      which = 1;
      status = 0;
      shape = (double) numsum;
      x = power;
      /* Determine the basic probability */
      cdfgam(&which, &p, &q, &x, &shape, &scale, &status, &bound);
      if (status) {
         fprintf(stderr, "\nError in cdfgam() (candidate_sigma()):\n");
         fprintf(stderr, "   status = %d, bound = %g\n", status, bound);
         fprintf(stderr, "   p = %g, q = %g, x = %g, shape = %g, scale = %g\n\n",
                p, q, x, shape, scale);
         exit(1);
      }
      /* Adjust it for the number of trials */
      if (p == 1.0)
         q *= numtrials;
      else
         q = 1.0 - pow(p, numtrials);
      p = 1.0 - q;
      which = 2;
      status = 0;
      /* Convert to a sigma */
      cdfnor(&which, &p, &q, &x, &mean, &sd, &status, &bound);
      if (status) {
         if (status == -2) {
            x = 0.0;
         } else if (status == -3) {
            x = 38.5;
         } else {
            fprintf(stderr, "\nError in cdfnor() (candidate_sigma()):\n");
            fprintf(stderr, "   status = %d, bound = %g\n", status, bound);
            fprintf(stderr, "   p = %g, q = %g, x = %g, mean = %g, sd = %g\n\n",
                   p, q, x, mean, sd);
            exit(1);
         }
      }
   }
   if (x < 0.0)
      return 0.0;
   else
      return x;
}

double power_for_sigma(double sigma, int numsum, double numtrials)
/* Return the approximate summed power level required */
/* to get a Gaussian significance of 'sigma', taking  */
/* into account the number of independent trials.     */
{
   int which, status;
   double p, q, x, bound, mean = 0.0, sd = 1.0, df, scale = 1.0;

   which = 1;
   status = 0;
   x = sigma;
   cdfnor(&which, &p, &q, &x, &mean, &sd, &status, &bound);
   if (status) {
      fprintf(stderr, "\nError in cdfnor() (power_for_sigma()):\n");
      fprintf(stderr, "   cdfstatus = %d, bound = %g\n\n", status, bound);
      fprintf(stderr, "   p = %g, q = %g, x = %g, mean = %g, sd = %g\n\n", p, q, x, mean, sd);
      exit(1);
   }
   q = q / numtrials;
   p = 1.0 - q;
   which = 2;
   df = 2.0 * numsum;
   status = 0;
   cdfchi(&which, &p, &q, &x, &df, &status, &bound);
   if (status) {
      fprintf(stderr, "\nError in cdfchi() (power_for_sigma()):\n");
      fprintf(stderr, "   status = %d, bound = %g\n", status, bound);
      fprintf(stderr, "   p = %g, q = %g, x = %g, df = %g, scale = %g\n\n",
             p, q, x, df, scale);
      exit(1);
   }
   return 0.5 * x;
}


double chisqr(double *data, int numdata, double avg, double var)
/* Calculates the chi-square of the 'data' which has average */
/* 'avg', and variance 'var'.                                */
{
   double dtmp, chitmp, chixmeas = 0.0;
   int ii;

   for (ii = 0; ii < numdata; ii++) {
      dtmp = data[ii];
      chitmp = dtmp - avg;
      chixmeas += (chitmp * chitmp);
   }
   return chixmeas / var;
}

void switch_f_and_p(double in, double ind, double indd,
                    double *out, double *outd, double *outdd)
/* Convert p, p-dot, and p-dotdot into f, f-dot, */
/* and f-dotdot or vise-versa.                   */
{
   double dtmp;

   *out = 1.0 / in;
   dtmp = in * in;
   if (ind == 0.0)
      *outd = 0.0;
   else
      *outd = -ind / dtmp;
   if (indd == 0.0)
      *outdd = 0.0;
   else
      *outdd = 2.0 * ind * ind / (dtmp * in) - indd / dtmp;
}


/* Optional non-macro definitions of power and phase */
/*

__inline__ double POWER(double rl, double im)
{
  return rl * rl + im * im;
}


__inline__ double PHASE(double rl, double im)
{
  double temp;

  temp = 57.2957795130823208 * atan2(im, rl);
  return (temp > 0.0) ? temp : temp + 360.0;
}


double RADIAN_PHASE(double rl, double im)
{
  double temp;

  temp = atan2(im, rl);
  return (temp > 0.0) ? temp : temp + TWOPI;
}

*/
