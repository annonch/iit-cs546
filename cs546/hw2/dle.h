/* function prototypes */
void setup(int argc, char **argv);
void usage();
void run();
void print_data();
clock_t getTime();
float diffTime(clock_t t1, clock_t t2);
void sequential();
void parallel_pthreads();
void parallel_openMP();
void guass_seq();
void guass_pthreads();
void guass_pthreads2();
void guass_openMP();
void back_sub();
void print_result();
void *poutine(void* pthread_data);
void *poutine2(void* pthread_data);
