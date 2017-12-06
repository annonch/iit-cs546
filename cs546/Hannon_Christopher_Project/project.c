#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#include "project.h"
#include "fft.c"



int MY_ID; // same as rank
int NUM_PROCS;

int MODE;
int MPI;

complex A[512][512];
complex A_i[512][512];
complex B[512][512];
complex B_i[512][512];

complex C[512][512];
complex D[512][512];
complex D_i[512][512];


int main(int argc, char **argv) 
{

  int ierr;
  float totalTime;

  setup(argc,argv);
  if (MY_ID ==0){
    printf("\n\n\n-------------------------------------------------------------------");
    printf("\ncs 546 Term Project\n");
    printf("Author: Christopher Hannon\n");
    
    printf("\n\tInitializing\n");
    printf("\n\tStarting Timer\n");
  }
  start_exec_timer();
  if(MY_ID==0)
    printf("\n\tRunning...\n");

  switch(MODE){
  case 1:
    if(MY_ID==0)
      printf("\nrunning ALGO 1\n");
    ALGO1();
    break;
  case 2:
    if(MY_ID==0)
      printf("\nrunning ALGO 2\n");
    ALGO2();
    break;
  case 3:
    if(MY_ID==0)
      printf("\nrunning ALGO 3\n");
    ALGO3();
    break;
  case 4:
    if(MY_ID==0)
      printf("\nrunning ALGO 4 (serial)\n");
    serial();
    break;
  }
  
  totalTime = print_exec_timer();
  if(MY_ID==0){
    printf("\n\tCleaning up..\n");
    printf("\n\tResults:\n");
    printf("\nTotal Time Spent: %15.6f s\n", totalTime);
    print_results();
    printf("-------------------------------------------------------------------\n\n\n");
  }
  cleanup();

  return 0;
}

void usage(){
  printf("\nUsage: mpirun -np 8 ./project ALGO \n");
  printf("Where ALGO = 1, 2, 3, or 4\n");
  printf("algo 1 is part a\n");
  printf("algo 2 is part b\n");
  printf("algo 3 is part c\n");
  printf("algo 4 is a serial implementation (No parallelism)\n");
}

void transpose(complex a[512][512]){
  int i,j;
  complex A_i[512][512];
  for(i=0;i<512;i++)
    for(j=0;j<512;j++)
      A_i[j][i]=a[i][j];
  a=A_i;
}

void serial(){ 
  /* Serial version */
  int i,j;
  c_fft1d(*A, 512, -1);
  c_fft1d(*B, 512, -1);
  transpose(A);
  transpose(B);
  c_fft1d(*A, 512, -1);
  c_fft1d(*B, 512, -1);
  for(i=0;i<512;i++)
    for(j=0;j<512;j++){
      D[i][j].r=A[i][j].r*B[i][j].r;
      D[i][j].i=A[i][j].i*B[i][j].i;
    }
  c_fft1d(*D,512,1);
  transpose(D);
  c_fft1d(*D,512,1);
}

void ALGO1() {
  MPI_Status Stat;
  int i,j;
  int range;

  range = 512/NUM_PROCS;
  
  
  // do first part of work
  c_fft1d((complex *)&A[range*MY_ID], range, -1);
  c_fft1d((complex *)&B[range*MY_ID], range, -1);


  if (MY_ID ==0){
    // master proc //
    // recv work
    for(i=1;i<NUM_PROCS;i++){
      MPI_Recv(&A[i*range], range*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD, &Stat);
      MPI_Recv(&B[i*range], range*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD, &Stat);
    }
    
    transpose(A);
    transpose(B);
    
    // send transposed
    for(i=1;i<NUM_PROCS;i++){
      MPI_Send(&A[i*range], range*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD);
      MPI_Send(&B[i*range], range*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD);
    }
  }
  else {
    //send work to proc 1
    MPI_Send(&A[MY_ID*range], range*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&B[MY_ID*range], range*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD);
    
    //recv transposed
    MPI_Recv(&A[MY_ID*range], range*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &Stat);
    MPI_Recv(&B[MY_ID*range], range*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &Stat);
  }
  
  //do second part of work
  c_fft1d((complex *)&A[range*MY_ID], range, -1);
  c_fft1d((complex *)&B[range*MY_ID], range, -1);

  //do all to all share

  if (MY_ID ==0){
    // master proc //
    // recv work
    for(i=1;i<NUM_PROCS;i++){
      MPI_Recv(&A[i*range], range*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD, &Stat);
      MPI_Recv(&B[i*range], range*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD, &Stat);
      }
    
    // send full Matrix
    for(i=1;i<NUM_PROCS;i++){
      MPI_Send(&A[0], 512*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD);
      MPI_Send(&B[0], 512*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD);
    }
  }
  else {
    //send work to proc 1
    MPI_Send(&A[MY_ID*range], range*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&B[MY_ID*range], range*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD);

    //recv full Matrix
    MPI_Recv(&A[0], 512*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &Stat);
    MPI_Recv(&B[0], 512*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &Stat);
  }
  
  //do task 3 work
  for(i=0;i<512;i++) {
    for(j=MY_ID*range;j<MY_ID*range+range;j++){
      D[i][j].r=A[i][j].r*B[i][j].r;
      D[i][j].i=A[i][j].i*B[i][j].i;
    }
  }

  //do all to all share
  if (MY_ID ==0){
    // master proc //
    // recv work
    for(i=1;i<NUM_PROCS;i++){
      MPI_Recv(&D[i*range], range*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD, &Stat);
    }
    
    // send full Matrix
    for(i=1;i<NUM_PROCS;i++){
      MPI_Send(&D[0], 512*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD);
    }
  }
  else {
    //send work to proc 1
    MPI_Send(&D[MY_ID*range], range*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD);
    
    //recv full Matrix
    MPI_Recv(&D[0], 512*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &Stat);
  }
  
  //do first part of task 4
  c_fft1d((complex *)&D[range*MY_ID], range, -1);
    
  if (MY_ID ==0){
    // master proc //
    // recv work
    for(i=1;i<NUM_PROCS;i++){
      MPI_Recv(&D[i*range], range*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD, &Stat);
  
    }
    
    transpose(D);
  
    // send transposed
    for(i=1;i<NUM_PROCS;i++){
      MPI_Send(&D[i*range], range*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD);

    }
  }
  else {
    //send work to proc 1
    MPI_Send(&D[MY_ID*range], range*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD);

    //recv transposed
    MPI_Recv(&D[MY_ID*range], range*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &Stat);
  }

  
  // do second part of task 4
  c_fft1d((complex *)&D[range*MY_ID], range, -1);
  
  if(MY_ID==0){
    for(i=1;i<NUM_PROCS;i++){
      MPI_Recv(&D[i*range], range*512, MPI_C_COMPLEX, i, 0, MPI_COMM_WORLD, &Stat); 
    }    
  }
  else {
    //send work to proc 1
    MPI_Send(&D[MY_ID*range], range*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}


/*
MPI_Allgather(
    void* send_data,
    int send_count,
    MPI_Datatype send_datatype,
    void* recv_data,
    int recv_count,
    MPI_Datatype recv_datatype,
    MPI_Comm communicator)

 */

void ALGO2() {
  MPI_Status Stat;
  int i,j;
  int range;
  

  // do T1 and T2 part A
  c_fft1d((complex *)&A[range*MY_ID], range, -1);
  c_fft1d((complex *)&B[range*MY_ID], range, -1);
 
  // all to all gather
  MPI_Allgather(&A[MY_ID*range], range*512, MPI_C_COMPLEX, &A_i[0], (range)*512, MPI_C_COMPLEX, MPI_COMM_WORLD);
  MPI_Allgather(&B[MY_ID*range], range*512, MPI_C_COMPLEX, &B_i[0], (range)*512, MPI_C_COMPLEX, MPI_COMM_WORLD);

  //printf("hello\n");
  
  //all transpose
  transpose(A_i);
  transpose(B_i);
  
  //do T1 and T2 part B
  c_fft1d((complex *)&A_i[range*MY_ID], range, -1);
  c_fft1d((complex *)&B_i[range*MY_ID], range, -1);
  
  // all to all gather
  MPI_Allgather(&A_i[MY_ID*range], range*512, MPI_C_COMPLEX, &A[0], (range)*512, MPI_C_COMPLEX, MPI_COMM_WORLD);
  MPI_Allgather(&B_i[MY_ID*range], range*512, MPI_C_COMPLEX, &B[0], (range)*512, MPI_C_COMPLEX, MPI_COMM_WORLD);

  //do T3
  for(i=0;i<512;i++) {
    for(j=MY_ID*range;j<MY_ID*range+range;j++){
      D_i[i][j].r=A[i][j].r*B[i][j].r;
      D_i[i][j].i=A[i][j].i*B[i][j].i;
    }
  }

  //all to all gather
  MPI_Allgather(&D_i[MY_ID*range], range*512, MPI_C_COMPLEX, &D[0], (range)*512, MPI_C_COMPLEX, MPI_COMM_WORLD);
  
  //do T4 part A
  c_fft1d((complex *)&D[range*MY_ID], range, -1);
  
  // all to all gather
  MPI_Allgather(&D[MY_ID*range], range*512, MPI_C_COMPLEX, &D_i[0], (range)*512, MPI_C_COMPLEX, MPI_COMM_WORLD);
  
  // do transpose
  transpose(D_i);

  //do T4 part B
  c_fft1d((complex *)&D_i[range*MY_ID], range, -1);
  
  //all to all gather
  MPI_Allgather(&D_i[MY_ID*range], range*512, MPI_C_COMPLEX, &D[0], (range)*512, MPI_C_COMPLEX, MPI_COMM_WORLD); 
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

void ALGO3() {
  /* Task and Data Parallel Model */
  // P1 = P2 = P3 = P4 = 2
  //T1 and T2 concurrently
  MPI_Status Stat;
  int i,j;

  // P1
  if(MY_ID==0){
    //compute half array A
    c_fft1d((complex *)&A[0], 256, -1);

    // share with mpi1
    MPI_Recv(&A[256], 256*512, MPI_C_COMPLEX, 1, 0, MPI_COMM_WORLD, &Stat);
    MPI_Send(&A[0], 256*512, MPI_C_COMPLEX, 1, 0, MPI_COMM_WORLD);

    //transpose
    transpose(A);

    //compute half array c A
    c_fft1d((complex *)&A[0], 256, -1);
      
    // share with mpi1
    MPI_Recv(&A[256], 256*512, MPI_C_COMPLEX, 1, 0, MPI_COMM_WORLD, &Stat);
    MPI_Send(&A[0], 256*512, MPI_C_COMPLEX, 1, 0, MPI_COMM_WORLD);

    //signal p3 okay to start
    MPI_Send(&A[0], 512*512, MPI_C_COMPLEX, 4, 0, MPI_COMM_WORLD);
    MPI_Send(&A[0], 512*512, MPI_C_COMPLEX, 5, 0, MPI_COMM_WORLD);

    //wait for final
    MPI_Recv(&D[0], 512*512, MPI_C_COMPLEX, 7,0,MPI_COMM_WORLD, &Stat);
    
  }
  else if(MY_ID==1){
    //compute half array A
    c_fft1d((complex *)&A[256], 256, -1);

    // share with mpi0
    MPI_Send(&A[256], 256*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(&A[0], 256*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &Stat);
    
    //transpose
    transpose(A);
    
    //compute half array c A
    c_fft1d((complex *)&A[256], 256, -1);
      
    // share with mpi0
    MPI_Send(&A[256], 256*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(&A[0], 256*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &Stat);
    
    //signal p3 okay to start
    
  }


  // P2
  else if(MY_ID==2){
    //compute half array A
    c_fft1d((complex *)&B[0], 256, -1);

    // share with mpi1
    MPI_Recv(&B[256], 256*512, MPI_C_COMPLEX, 3, 0, MPI_COMM_WORLD, &Stat);
    MPI_Send(&B[0], 256*512, MPI_C_COMPLEX, 3, 0, MPI_COMM_WORLD);

    //transpose
    transpose(B);

    //compute half array c A
    c_fft1d((complex *)&B[0], 256, -1);
      
    // share with mpi1
    MPI_Recv(&B[256], 256*512, MPI_C_COMPLEX, 3, 0, MPI_COMM_WORLD, &Stat);
    MPI_Send(&B[0], 256*512, MPI_C_COMPLEX, 3, 0, MPI_COMM_WORLD);

    //signal p3 okay to start
    MPI_Send(&B[0], 512*512, MPI_C_COMPLEX, 4, 0, MPI_COMM_WORLD);
    MPI_Send(&B[0], 512*512, MPI_C_COMPLEX, 5, 0, MPI_COMM_WORLD);
    
  }
  else if(MY_ID==3){
    //compute half array A
    c_fft1d((complex *)&B[256], 256, -1);

    // share with mpi0
    MPI_Send(&B[256], 256*512, MPI_C_COMPLEX, 2, 0, MPI_COMM_WORLD);
    MPI_Recv(&B[0], 256*512, MPI_C_COMPLEX, 2, 0, MPI_COMM_WORLD, &Stat);
    
    //transpose
    transpose(B);
    
    //compute half array c A
    c_fft1d((complex *)&B[256], 256, -1);
      
    // share with mpi0
    MPI_Send(&B[256], 256*512, MPI_C_COMPLEX, 2, 0, MPI_COMM_WORLD);
    MPI_Recv(&B[0], 256*512, MPI_C_COMPLEX, 2, 0, MPI_COMM_WORLD, &Stat);
    
    //signal p3 okay to start
    
  }

  //P3
  else if(MY_ID==4){
    // wait till p1 and p2 finish
    MPI_Recv(&A[0], 512*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &Stat);
    MPI_Recv(&B[0], 512*512, MPI_C_COMPLEX, 2, 0, MPI_COMM_WORLD, &Stat);
    // Do top half
    for(i=0;i<512;i++){
      for(j=0;j<256;j++){
	C[j][i].r=A[i][j].r * B[i][j].r ;
	C[j][i].i=A[i][j].i * B[i][j].i ;
      }
    }
    MPI_Recv(&C[256], 256*512, MPI_C_COMPLEX, 5, 0, MPI_COMM_WORLD, &Stat);
    MPI_Send(&C[0], 256*512, MPI_C_COMPLEX, 5, 0, MPI_COMM_WORLD);

    MPI_Send(&C[0], 512*512, MPI_C_COMPLEX, 6 ,0 ,MPI_COMM_WORLD);
    MPI_Send(&C[0], 512*512, MPI_C_COMPLEX, 7 ,0 ,MPI_COMM_WORLD);

  }
  else if(MY_ID==5){
    // wait till p1 and p2 finish
    MPI_Recv(&A[0], 512*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &Stat);
    MPI_Recv(&B[0], 512*512, MPI_C_COMPLEX, 2, 0, MPI_COMM_WORLD, &Stat);
    // Do bottom half
    for(i=0;i<512;i++){
      for(j=256;j<512;j++){
	C[j][i].r=A[i][j].r * B[i][j].r ;
	C[j][i].i=A[i][j].i * B[i][j].i ;
      }
    }
    MPI_Send(&C[256], 256*512, MPI_C_COMPLEX, 4, 0, MPI_COMM_WORLD);
    MPI_Recv(&C[0], 256*512, MPI_C_COMPLEX, 4, 0, MPI_COMM_WORLD, &Stat);

    
  }


  //P4
  else if(MY_ID==6){
    //wait till p3 finish
    MPI_Recv(&D[0], 512*512, MPI_C_COMPLEX,4 ,0 ,MPI_COMM_WORLD, &Stat);

    //compute half array A
    c_fft1d((complex *)&D[0], 256, -1);

    // share with mpi1
    MPI_Recv(&D[256], 256*512, MPI_C_COMPLEX, 7, 0, MPI_COMM_WORLD, &Stat);
    MPI_Send(&D[0], 256*512, MPI_C_COMPLEX, 7, 0, MPI_COMM_WORLD);

    //transpose
    transpose(D);

    //compute half array c A
    c_fft1d((complex *)&D[0], 256, -1);
      
    // share with mpi1
    MPI_Recv(&D[256], 256*512, MPI_C_COMPLEX, 7, 0, MPI_COMM_WORLD, &Stat);
    MPI_Send(&D[0], 256*512, MPI_C_COMPLEX, 7, 0, MPI_COMM_WORLD);
   
    
  }
  else { // 7
    //wait till p3 finish
    MPI_Recv(&D[0], 512*512, MPI_C_COMPLEX,4 ,0 ,MPI_COMM_WORLD, &Stat);

    //compute half array A
    c_fft1d((complex *)&D[256], 256, 1);

    // share with mpi0
    MPI_Send(&D[256], 256*512, MPI_C_COMPLEX, 6, 0, MPI_COMM_WORLD);
    MPI_Recv(&D[0], 256*512, MPI_C_COMPLEX, 6, 0, MPI_COMM_WORLD, &Stat);
    
    //transpose
    transpose(D);
    
    //compute half array c A
    c_fft1d((complex *)&D[256], 256, 1);
      
    // share with mpi0
    MPI_Send(&D[256], 256*512, MPI_C_COMPLEX, 6, 0, MPI_COMM_WORLD);
    MPI_Recv(&D[0], 256*512, MPI_C_COMPLEX, 6, 0, MPI_COMM_WORLD, &Stat);
    
    MPI_Send(&D[0],512*512, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD);
    
  }
  MPI_Barrier(MPI_COMM_WORLD);

}

void setup(int argc, char **argv){
  /* sets up the random numbers*/
  int ierr, my_id, num_procs;
  FILE *f_A, *f_B; // open file discriptor
  int i,j;

  my_id=0;
  
  if(argc<2){
    printf("incorrect args\n");
    usage();
    exit(1); // incorrect args
  }

  MODE = atoi(argv[1]);
  //printf("%d\n",MODE);
  if(MODE >4 || MODE <1){//!= 1 || MODE != 2 || MODE != 3 || MODE != 4) {
    printf("MODE ERROR\n");
    usage();
    exit(1);
  }
  if (MODE < 4)
    MPI=1;
  else
    MPI=0;
   
  /* MPI SETUP */
  if (MPI){
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if(MODE == 3 && num_procs != 8){
      printf("Error, number of processes for ALGO 3 need to be 8\n");
      exit(2);
    }
    if(!(num_procs == 1 || num_procs == 2 || num_procs == 4 || num_procs == 8)){
      printf("Error, num processors specified must be 1,2,4,or 8\n");
      exit(2);
    }
    if (num_procs==1)
      MODE=4;
    NUM_PROCS=num_procs;
    MY_ID=my_id;
  }

  // read inputs from https://www.programiz.com/c-programming/examples/read-file
  if ((f_A = fopen("im1", "r")) == NULL) {
      printf("Error! opening file A");
      // Program exits if file pointer returns NULL.
      exit(3);         
  }
  if ((f_B = fopen("im2", "r")) == NULL) {
      printf("Error! opening file B");
      // Program exits if file pointer returns NULL.
      exit(3);         
  }
 
  for(i=0;i<512;i++) 
    for(j=0;j<512;j++)
      fscanf(f_A,"%g",&A[i][j]);
  for(i=0;i<512;i++) 
    for(j=0;j<512;j++)
      fscanf(f_B,"%g",&B[i][j]);

  fclose(f_A);
  fclose(f_B);
}



void cleanup(){
  if (MPI){
    MPI_Barrier(MPI_COMM_WORLD);
    int ierr;
    ierr = MPI_Finalize();
  }
}

void print_results(){
  FILE *f_o;
  int i,j;

  if ((f_o = fopen("out", "w")) == NULL) {
      printf("Error! opening file out");
      // Program exits if file pointer returns NULL.
      exit(3);         
  }

  
  for(i=0;i<512;i++){
    for(j=0;j<512;j++){
      fprintf(f_o,"%6.2g ",&D[i][j]);
    }
    fprintf(f_o,"\n");
  }
  fclose(f_o);
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
/*
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
