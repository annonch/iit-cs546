#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <math.h>

//#include <GL/glut.h>

//#include <cutil_inline.h>
//#include <cuda.h>


/* This code is adapted from https://devtalk.nvidia.com/default/topic/453819/cuda-programming-and-performance/the-game-of-life-in-cuda/2
 * The authors use openGL
 * Here we print an asci grid instead
 *
 *  our default size is 1000
 */



int* H_a; // for serial (frame a)
int* H_b; // for serial (frame b)  esssentially a temparary buffer for the updates from the previous interatiomn/cycle/generation
int* D_a;
int* D_b;


// define the size of the world 
#define SCREENX 1000
#define SCREENY 1000

//how far to print the ascii minitor
#define PRINTVAL 10


// blocksize
#define XBLOCKSIZE 16;
#define YBLOCKSIZE 16;


//changing this number is really cool in opengl but not so much in the ascii print

float POPULATION=0.3125; //Chance, that the Random Starting Population generator decides to create a new individual
//float POPULATION=0.062125; //Chance, that the Random Starting Population generator decides to create a new individual

int GetIndividual(int x, int y, int* Array) {
  /* Get individual returns the value of an (x,y) coordinate from the equivilent 1D array mapping */
  return (Array[x+(SCREENX*y)]);
}

void SetIndividual(int x, int y, int val, int* Array) {
  /* Set individual sets the value of an (x,y) coordinate from the equvilent 1D array mapping */
	Array[x+(SCREENX*y)]=val;
}

int Neighbors(int x, int y, int* Array) {
  /* The rules dictate that the 8 neighbors should be checked for life */

  /*
    o x o
    x A o
    o o o 

    In the above example A is the (x,y) address of the element looked at and the result is = 2
   */
  
  int i, k, anz=0;
  
  for (i=-1;i<=1;i++)
    for (k=-1;k<=1;k++) {
	if (!((i==0)&&(k==0)) && (x+i<SCREENX) && (y+k<SCREENY) && (x+i>0) && (y+k>0)) {
	    if (GetIndividual(x+i, y+k, Array)>0)
	      anz++;
	  }
      }
  return anz;
}

void SpawnPopulation(float frequenzy, int* Array) {
  /* This function sets the initial population of the world with the probability set above in POPULATION */
  int random, x,y;
  //seed random with current time
  srand ( time(NULL) );  
  for (x=0;x<SCREENX;x++)
    for (y=0;y<SCREENY;y++)
      {
	random=rand() % 100;
	if ((float)random/100.>frequenzy)
	  SetIndividual(x,y,0, Array);
	else 
	  SetIndividual(x,y,1, Array); 
      }
}

void NextGeneration() {
  /* cycles through world serially setting the second buffer alive or dead appropriately */
  int x, y, n, a;
  for (x=1;x<SCREENX;x++)
    for (y=1;y<SCREENY;y++) {
	n=Neighbors(x,y,H_a);
	a=GetIndividual(x,y,H_a);
	
	if (a>0) {
	    if ((n>3) || (n<2))
	      SetIndividual(x,y,0, H_b);
	    else
	      SetIndividual(x,y,1, H_b);
	  }
	else {//if (GetIndividual(x,y,H_a)==0) {
	    if (n==3)
	      SetIndividual(x,y,1, H_b);
	    else
	      SetIndividual(x,y,0, H_b);  
	  }
      }
  // copy buffer back to origional 
  memcpy(H_a, H_b, SCREENX*SCREENY*sizeof(int)); 
}


void printWorld() {
  int i,j;
  for(i=0;i<PRINTVAL;i++){
    for(j=0;j<PRINTVAL;j++){
      if (GetIndividual(i,j,H_a)) { //H_a[i,j]==1){
	  printf("x ");
	}
      else {
	printf("o ");
      }
    }
    printf("\n");
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
    int nsec = (end.tv_usec - start.tv_usec) / 1000000;
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




int main(int argc, char **argv)
{
  int idev, deviceCount;
  int i;
  float totalTime, nextStepTime;
  totalTime=0.0;
  
  char *device = NULL;

  //serial run

  H_a=(int*)malloc(SCREENX*SCREENY*sizeof(int));                                                                                                        
  H_b=(int*)malloc(SCREENX*SCREENY*sizeof(int));                                                                                                        

  //init the world
  SpawnPopulation(POPULATION, H_a);               

  printf("initial world (corner)\n");
  printWorld();
  printf("starting the timer\n");


  start_exec_timer();
  for(i=0;i<10;i++) {
    NextGeneration();
  }
  printf("after 10 cycles\n");
  nextStepTime = print_exec_timer();
  printWorld();
  totalTime += nextStepTime;
  printf("10 cycles: %f time/cycle %f\n",nextStepTime,totalTime/10.0 );
  
  
  start_exec_timer();
  for(;i<100;i++) {
    NextGeneration();
  }
  printf("after 100 cycles\n");
  nextStepTime = print_exec_timer();
  printWorld();
  totalTime += nextStepTime;
  printf("100 cycles: %f time/cycle %f\n",nextStepTime,totalTime/100.0 );

  
  start_exec_timer();
  for(;i<1000;i++) {
    NextGeneration();
  }
  printf("after 1000 cycles\n");
  nextStepTime = print_exec_timer();
  printWorld();
  totalTime += nextStepTime;
  printf("1000 cycles: %f time/cycle %f\n",nextStepTime,totalTime/1000.0 );
  
}


