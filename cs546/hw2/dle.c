/* Author: Christopher Hannon
 * cs546 Parallel and Distributed Processing
 * Homework 2
 * Shared Memory Programming
 *
 * This Program implements 3 algorithms for solving
 * dense linear equations of the
 * form A*x=b, where A is an n*n matrix and b is a
 * vector. This program performs guassian elimination
 * without pivoting and back substitution.
 *
 * The 3 algorithms are sequential, parallel - Pthreads
 * and parallel - OpenMP
 *
 * In the Parallel implementations (pthreads/openMP),
 * The guassian elimination is performed in parallel
 * while back substitution is performed sequentially.
 * Normalization is done in the back substitution..
 * i.e., diagionals are not normalized.
 *
 * ALGORITHM EXPLAINATIONS:
 *
 * Sequential:
 *   This algorithm is simple and consists of 3 for loops
 *   bounded by N.
 *
 * Pthreads:
 *   This algorithm is explained in the comments above the
 *   implementation function and in the PDF.
 *
 *
 * OpenMP:
 *
 *
 *
 *
 * * Start Wolfram Mathworld Quote
 * * http://mathworld.wolfram.com/GaussianElimination.html
 *
 * Guassian elimination is a method for solving matrix
 * equations of the form: Ax=b
 *
 * To perform Guassian elimination starting with the system
 * of equations:
 *  __             __  _  _       _  _
 * |                 ||    |     |    |
 * | a11 a12 ... a1k || x1 |     | b1 |
 * | a21 a22 ... a2k || x2 |  =  | b2 |
 * | ............... || .. |     | .. |
 * | ak1 ak2 ... akk || xk |     | bk |
 * |__             __||_  _|     |_  _|
 *
 * compose the "augmented matrix equation"
 *  __                  __  _  _
 * |                      ||    |
 * | a11 a12 ... a1k | b1 || x1 |
 * | a21 a22 ... a2k | b2 || x2 |
 * | ............... | .. || .. |
 * | ak1 ak2 ... akk | bk || xk |
 * |__                  __||_  _|
 *
 * Here, the column vector in the variables 'x' is carried
 * along for labeling the matrix rows. Now, perform elementary
 * row operations to put the augmented matrix into the upper
 * triangular form
 *  __                      __
 * |                          |
 * | a'11 a'12 ... a1'k | b'1 |
 * |  0   a'22 ... a2'k | b'2 |
 * |  ................  | ... |
 * |  0    0   ... a'kk | b'k |
 * |__                      __|
 *
 * Solve the equation of the kth row for xk, then substitute
 * back into the equation of the (k-1)st row to obtain a
 * solution for x(k-1), etc., according to the formula:
 *
 * xi = 1/a'ii( b'i - {\sum ^k _ (j=i+1)} a'ij*xj)
 *
 * * End Wolfram Mathworld Quote
 * * http://mathworld.wolfram.com/GaussianElimination.html
 *
 *
 *
 * Exit Codes:
 *  0 - Program executed successfully
 * -1 - Incorrect arguments to program (see Usage)
 * -2 - failed on pthreads
 * -3 - failed on semaphores
 * -4 - failed on openMP
 *
 * Usage:
 *  ./guass (0/1/2/3) (N)
 *   0 - Sequential mode
 *   1 - Pthreads mode
 *   2 - OpenMP mode
 *   3 - Test all (Not implemented)
 */


/* includes */
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <omp.h>
#include "dle.h"

/* arbitrarily choose max n of 1000 */
#define MAX_SIZE_OF_N 1000
/* arbitrarily choose max size */
#define MAX_SIZE_ELEMENT 10000

float A[MAX_SIZE_OF_N][MAX_SIZE_OF_N];
float B[MAX_SIZE_OF_N];
float X[MAX_SIZE_OF_N];

int N;

sem_t Locks[MAX_SIZE_OF_N];

/* Pthreads struct for holding i and j */
struct pthread_data {
  int i;
  int j;
};

struct pthread_data pthread_data_array[MAX_SIZE_OF_N];

void setup(int argc, char **argv) {
  int i,j;
  if (argc < 3) {
    usage();
    exit(1); // error code 1 = incorrect args
  }
  N = atoi(argv[2]);
  /* Check input data */
  if (N > 1000) {
    printf("Max size of array: 1000 x 1000\n");
    N = 1000;
  }
  if (N < 1) {
    printf("Min size of array: 1 x 1\n");
    N = 1;
  }

  /* create data */
  srand(time(NULL));
  for(i=0; i<N; i++) {
    for(j=0; j<N; j++) {
      A[i][j]=((double)rand() /((double)RAND_MAX + 1) * MAX_SIZE_ELEMENT);
    }
    B[i] = ((double)rand() /((double)RAND_MAX + 1) * MAX_SIZE_ELEMENT);
  }
}

void run(char **argv) {
  switch(atoi(argv[1])) {
  case 0 :
    /* sequential */
    sequential();
    break;
  case 1 :
    /* Pthreads */
    parallel_pthreads();
    break;
  case 2 :
    /* OpenMP */
    parallel_openMP();
    break;
  case 3 :
    /* Test All */
    printf("Not implemented.");
    break;
  default :
    printf("Invalid argument for test\n");
    usage();
    exit(1);
  }
}

void usage() {
    printf("\nUsage: ./guass (0/1/2/3) (N)\n");
    printf("0 - sequential mode\n");
    printf("1 - Pthreads mode\n");
    printf("2 - OpenMP mode\n");
    printf("3 - Test all\n");
    printf("N - size of A array\n\n");
}

void print_result() {
  int i;
  printf("\n\tResult: X = \n");
  printf(" _          _ \n");
  printf("|            |\n");

  for(i=0; i<N; i++) {
    printf("| %10.2f |\n", X[i]);
  }
  printf("|_          _|\n");
}

void print_data() {
  /* print data beautifully */
  int i,j;
  if (N > 20) {
    return;
  }
  /* print top */
  printf(" __");
  for (i=0; i< N; i++) {
    printf("           "); // 8 spaces
  }
  printf("           __\n");
  printf("|   ");
  for (i=0; i< N; i++) {
    printf("           "); // 8 spaces
  }
  printf("             |\n");
  /* print data */
  for (j=0; j<N; j++){ //row
    printf("|   ");
    for (i=0; i<N; i++) {
      printf("%10.2f ",A[j][i]);
    }
    printf("| %10.2f |\n", B[j]);
  }
  /* print bottom */
  printf("|__");
  for (i=0; i< N; i++) {
    printf("           "); // 8 spaces
  }
  printf("            __|\n");
}

/* new clock */
/* timing code is from stackoverflow */
/* https://stackoverflow.com/questions/459691/best-timing-method-in-c */

int timeval_sub(struct timeval *result, struct timeval end, struct timeval start) {
  if (start.tv_usec < end.tv_usec) {
    int nsec = (end.tv_usec - start.tv_usec) / 1000000 + 1;
    end.tv_usec -= 1000000 * nsec;                                                                       end.tv_sec += nsec;                                                                                }                                                                                                    if (start.tv_usec - end.tv_usec > 1000000) {                                                           int nsec = (end.tv_usec - start.tv_usec) / 1000000;                                                  end.tv_usec += 1000000 * nsec;                                                                       end.tv_sec -= nsec;                                                                                }                                                                                                    result->tv_sec = end.tv_sec - start.tv_sec;                                                          result->tv_usec = end.tv_usec - start.tv_usec;
  return end.tv_sec < start.tv_sec;
}

float set_exec_time(int end) {
  static struct timeval time_start;
  struct timeval time_end;
  struct timeval time_diff;
  if (end) {
    gettimeofday(&time_end, NULL);
    if (timeval_sub(&time_diff, time_end, time_start) == 0) {
      if (end == 1)
	//printf("\nexec time: %1.2fs\n", time_diff.tv_sec + (time_diff.tv_usec / 1000000.0f));
	return time_diff.tv_sec + (time_diff.tv_usec / 1000000.0f);
      else if (end == 2)
	printf("%1.2fs",
	       time_diff.tv_sec + (time_diff.tv_usec / 1000000.0f));
    }
    return -1;
  }
  gettimeofday(&time_start, NULL);
  return 0;
}

void start_exec_timer() {
  set_exec_time(0);
}

float print_exec_timer() {
  return set_exec_time(1);
}


clock_t getTime() {
  return clock();
}

float diffTime(clock_t t1, clock_t t2) {
  return ((float)(t2 - t1) / (float)CLOCKS_PER_SEC ) * 1000;
}

/* Wrapper Functions */

void sequential() {
  printf("\tExecuting sequentially.\n");
  guass_seq();
  print_data();
  back_sub();
  print_result();
}

void parallel_pthreads() {
  printf("\tExecuting in Parallel using Pthreads.\n");
  guass_pthreads2();
  print_data();
  back_sub();
  print_result();
}

void parallel_openMP() {
  printf("\tExecuting in Parallel using openMP.\n");
  guass_openMP();
  print_data();
  back_sub();
  print_result();
}

void guass_seq() {
  int norm, row, col;
  float multiplier;

  /* Guassian Elimination */
  for (norm = 0; norm < N - 1; norm++) {
    for (row = norm + 1; row < N; row++) {
      multiplier = A[row][norm] / A[norm][norm];
      for (col = norm; col < N; col++) {
	A[row][col] -= A[norm][col] * multiplier;
      }
      B[row] -= B[norm] * multiplier;
    }
  }
}

void guass_pthreads2() {
  /* Parallel implementaition using Pthreads w/o rounds */
  int i,j;
  pthread_t threads[N];
  int rc;  //return code from pthread_create and sem_init

  /*Instead of using N-1 rounds, we embed the 'rounds'
   * into the pthreads routine. We use N-1 semaphores to protect
   * the columns that havent been reduced yet. we start with
   * all sems are 'locked' and then the sem_k is 'unlocked' by
   * pthread_k before it exits. */

  /* init and lock semaphores*/
  for(i=0; i<N; i++) {
    /* int sem_init(sem_t *sem, int pshared, unsigned int value);
     * calls can be sem_wait(), sem_try_wait(), sem_post() and sem_destroy() */
    rc = sem_init(&Locks[i],1,0); // create with value 0 and shared
    if(rc){
      printf("ERROR!!!!; sem_init failed with %d.\n",rc);
      exit(-3);
    }

  }
  for(j=1;j<N+1;j++) { //O(N^2/p)
    /* kickoff N-i-1 threads */
    pthread_data_array[j].i=0; // not needed
    pthread_data_array[j].j=j; //determines row

    rc = pthread_create(&threads[j], NULL, poutine2, (void *) &pthread_data_array[j]);
    if (rc){
      printf("Error!!!!!; pthread_create failed with %d\n", rc);
      exit(-2);
    }

  }
  sem_post(&Locks[0]);
  for(j=1;j<N+1;j++) {
    /* join all pthreads */
    pthread_join(threads[j],NULL);
  }
}

void *poutine2 (void *pthreadarg) {
  /* A Quebecois pthread routine */
  int i,j,col,isLocked;
  float mult;
  struct pthread_data *loc_data;
  loc_data = (struct pthread_data *) pthreadarg;
  i = loc_data->i;
  j = loc_data->j;  //myrow

  /* start at i and Use the i_th row to eliminate the i_th column
   * of the j_th row */
  for(i;i<j;i++){
    //see if sem is unlocked
    isLocked=0;
    while(!isLocked){
      sem_getvalue(&Locks[i],&isLocked);
    }
    /* modify A */
    mult = A[j][i]/A[i][i];
    for(col=i;col<N+1;col++) {
      A[j][col] -= mult * A[i][col];
    }
    B[j]-= B[i] * mult;
  }
  sem_post(&Locks[j]);
  /* can call pthread_exit here or else is implied */
  pthread_exit(NULL);

}

void guass_pthreads() {
  /* Parallel implementation using Pthreads */
  int i, j;
  pthread_t threads[N];
  int rc; //return code from pthread_create
  /*
   * we can have N-1 rounds where in each round
   * a column is eliminated, each row is ran
   * in parallel in a manager/worker paradigm
   *
   * x x x x x      x x x x x     x x x x x     x x x x x     x x x x x
   * x x x x x      o x x x x     o x x x x     o x x x x     o x x x x
   * x x x x x -->  o x x x x --> o o x x x --> o o x x x --> o o x x x
   * x x x x x      o x x x x     o o x x x     o o o x x     o o o x x
   * x x x x x      o x x x x     o o x x x     o o o x x     o o o o x
   *     0              1             2             3             4
   */
  for(i=0; i<N-1; i++) { //O(N)
    /* Rounds:
     * use the i_th row to to remove the i_th
     * column of the j_th row
     */
    for(j=i+1;j<N+1;j++) { //O(N^2/p)
      /* kickoff N-i-1 threads */
      pthread_data_array[j].i=i;
      pthread_data_array[j].j=j;

      rc = pthread_create(&threads[j], NULL, poutine, (void *) &pthread_data_array[j]);
      if (rc){
	printf("Error!!!!!; pthread_create failed with %d\n", rc);
	exit(-2);
      }

    }
    for(j=i+1;j<N+1;j++) {
      /* join all pthreads */
      pthread_join(threads[j],NULL);
    }
  }
}

void *poutine (void *pthreadarg) {
  /* A Quebecois pthread routine */
  int i,j,col;
  float mult;
  struct pthread_data *loc_data;
  loc_data = (struct pthread_data *) pthreadarg;
  i = loc_data->i;
  j = loc_data->j;
  /* Use the i_th row to eliminate the i_th column
   * of the j_th row */

  /* modify A */
  mult = A[j][i]/A[i][i];
  for(col=i;col<N+1;col++) {
    A[j][col] -= mult * A[i][col];
  }
  B[j]-= B[i] * mult;
  /* can call pthread_exit here or else is implied */
  pthread_exit(NULL);
}

void guass_openMP() {
  /* The OpenMP implementation is similar to the pthreads 1 implemenation */
  int i,j,col;
  float mult;
  
  for(i=0; i<N-1; i++) { //O(N)
    
    #pragma omp parallel num_threads(8)  default(shared) private(j,col,mult)
    /* Rounds:
     * use the i_th row to to remove the i_th
     * column of the j_th row
     */
    for(j=i+1;j<N+1;j++) { //O(N^2/p) 
      /* modify A */
      mult = A[j][i]/A[i][i];
      for(col=i;col<N+1;col++) {
	A[j][col] -= mult * A[i][col];
      }
      B[j]-= B[i] * mult;
    }
    #pragma omp barrier
    #pragma omp single
  }
}

void back_sub() {
  int row,col;

  /* Back Substitution */
  for (row = N-1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col --){
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}

int main(int argc, char **argv) {
  float totalTime;
  //clock_t startTime,endTime;
  /* Main routine */
  printf("\n\n\n---------------------------------------------------------------------------------------------------------------\n");
  printf("cs 546 HW 2 Shared Memory Programming.\n");
  printf("\nStep 1: initializing.\n");
  setup(argc,argv);
  printf("\tGenerated data: [A|b] =\n");
  print_data();
  printf("\tStarting Timer\n");
  //startTime=getTime();
  start_exec_timer();
  printf("\nStep 2: running...\n");
  run(argv);
  //endTime=getTime();
  totalTime = print_exec_timer();
  printf("\nTotal Time Spent: %15.6f s\n", totalTime);//diffTime(startTime,endTime));
  printf("\n---------------------------------------------------------------------------------------------------------------\n");
  /* exit program */
  return 0;
}
