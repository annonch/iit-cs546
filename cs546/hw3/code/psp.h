void usage();
void calc_seq();
void setup(int argc,char **argv);
void cleanup();
void printst(int s,int t);
void run();
int timeval_sub(struct timeval *result, struct timeval end, struct timeval start);
float set_exec_time(int end);
void start_exec_timer();
float print_exec_timer();

