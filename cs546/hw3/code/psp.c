#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>

#include "psp.h"


#define MAX_N 1048576 // 2^20
#define MAX_SIZE = 1024 // arbitrary

int N;
int P;
int RANK;
int MY_ID; // same as rank

int Ns[MAX_N]; 
int Fs[MAX_N]; 

main(int argc, char **argv) 
{

  int ierr;
  float totalTime;

  printf("\n\n\n-------------------------------------------------------------------");
  printf("cs 546 HW 3 MPI Programming\n");
  printf("\n\tInitializing\n");
  setup(argc,argv);
  printf("\tStarting Timer\n");
  start_exec_timer();
  printf("\n\tRunning...\n");
  run(argv);
  totalTime = print_exec_timer();
  printf("\n\tCleaning up..\n");
  printf("\nResults:\n");
  print_results();
  printf("\nTotal Time Spent: %15.6f s\n", totalTime);
  printf("-------------------------------------------------------------------\n\n\n");
  return 0;
}

void usage(){
  printf("\nUsage: ./pcp N P\n");
  printf("Where P = {1,2,4,8,16}\n");
  printf("And N mod P = 0");
}

void add_inher(int running){

  int i;
  int end;

  end = N/P;
  
  for(i=0;i<end;i++){

    Fs[MY_ID+i] = running;
  }
  return running;
}
  

}


int calc_seq(){ 
  /*each rank computes in seq before sharing*/
  // calc from myID to n/p+ myid

  int i;
  int running;
  int end;

  running = 0;
  end = N/P;
  
  for(i=0;i<end;i++){
    running+= Ns[MY_ID+i]
    Fs[MY_ID+i] = running;
  }
  return running;
}

void setup(int argc, char **argv){
  /* sets up the random numbers*/
  int i, ierr, my_id, num_procs;

  if(argc<3){
    usage();
    exit(1); // incorrect args
  }

  if(P != 1 || P != 2 || P != 4 || P != 8 || P != 16) {
    usage();
    exit(1);
  }
  if(N%P != 0){
    usage();
    exit(1)
  }
  
  srand(time(NULL));
  for(i=0;i<N;i++) {
    Ns[i]=((int)rand() / ((double)RAND_MAX +1) * MAX_SIZE);
  }
  
  /* MPI SETUP */ 
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  if(num_procs != P){
    printf("Error, number of processes specified not equal to P");
    exit(2);
  }
  MY_ID=my_id;
}

void cleanup(){
  int ierr;
  ierr = MPI_Finalize();
}

void print_results(){


}

void printst(int s, int t){
  printf("\n");
  for(s;s<t;s++){
    printf("%d\t"Fs[s]);
  }
  printf("\n");
}

void run() {
  int seq_tot, inher;


  seq_tot = run_seq();
  inher = do_MPI_stuff();
  add_inher(inher);
 
}


/*
MPI_Send(
    void* data,
    int count,
    MPI_Datatype datatype,
    int destination,
    int tag,
    MPI_Comm communicator)

MPI_Recv(
    void* data,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm communicator,
    MPI_Status* status)
*/

int do_MPI_stuff(int val){
  int sval;
  int s1,s2,s3,s4;
  int rec;
  
  if(P==2){
    if(MY_ID  == 0) { // even
      // recieve
      MPI_Recv(&rec,1,MPI_INT,1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      sval = rec + val;
      // return val
      MPI_Send(&sval,1,MPI_INT,1,1,MPI_COMM_WORLD);
      return val;
    }
    else{
      //sendval
      sval=val;
      MPI_Send(&sval,1,MPI_INT,0,1,MPI_COMM_WORLD);
      //recieve
      MPI_Recv(&rec,1,MPI_INT,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      return val+rec;
    }
  }
  
  else if(P == 4){
    if(MY_ID  == 0) { // even
      // recieve
      MPI_Recv(&rec,1,MPI_INT,1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      sval = rec + val;
      // return val
      MPI_Send(&sval,1,MPI_INT,1,1,MPI_COMM_WORLD);

      //talk to 2
      
      return val;
    }
    else if(MY_ID == 1{
      //sendval
      sval=val;
      MPI_Send(&sval,1,MPI_INT,0,1,MPI_COMM_WORLD);
      //recieve
      MPI_Recv(&rec,1,MPI_INT,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      return val+rec;
    }
    if(MY_ID == 2) { // even
      // recieve
      MPI_Recv(&rec,1,MPI_INT,3,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      sval = rec + val;
      // return val

      MPI_Send(&sval,1,MPI_INT,0,1,MPI_COMM_WORLD);
      MPI_Recv(&rec,1,MPI_INT,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      sval = sval + rec
      
      MPI_Send(&sval,1,MPI_INT,3,1,MPI_COMM_WORLD);
      return val;
    }
    else{ //3
      //sendval
      sval=val;
      MPI_Send(&sval,1,MPI_INT,2,1,MPI_COMM_WORLD);
      //recieve
      MPI_Recv(&rec,1,MPI_INT,2,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      return val+rec;
    }
  }
    
  else if(P == 8){
    
  }
  else{
    //16
  }
  
}

/* timing code is from stackoverflow */
/* https://stackoverflow.com/questions/459691/best-timing-method-in-c */

int timeval_sub(struct timeval *result, struct timeval end, struct timeval start) {
  if (start.tv_usec < end.tv_usec) {
    int nsec = (end.tv_usec - start.tv_usec) / 1000000 + 1;
    end.tv_usec -= 1000000 * nsec;
    end.tv_sec += nsec;
  }
  
  if (start.tv_usec - end.tv_usec > 1000000) {
    int nsec = (end.tv_usec - start.tv_usec)  / 1000000;
    end.tv_usec += 1000000 * nsec;					
    end.tv_sec -= nsec;							
  }									
  result->tv_sec = end.tv_sec - start.tv_sec;				
  result->tv_usec = end.tv_usec - start.tv_usec;
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
        printf("%1.2fs", time_diff.tv_sec + (time_diff.tv_usec / 1000000.0f));
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
