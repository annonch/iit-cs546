void usage();
void serial();
void ALGO1();
void ALGO2();
void ALGO3();

void setup(int argc,char **argv);
void cleanup();
//void printst(int s,int t);
//void run();
int timeval_sub(struct timeval *result, struct timeval end, struct timeval start);
float set_exec_time(int end);
void start_exec_timer();
float print_exec_timer();

//int do_MPI_stuff(int val);


void print_results();
//void transpose(complex a[512][512]);
