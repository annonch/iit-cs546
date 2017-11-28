#include <stdio.h>
#include <stdlib.h>

#include <GL/glut.h>

//#include <cutil_inline.h>
#include <cuda.h>

int* H_a;
int* H_b;
int* D_a;
int* D_b;

#define SCREENX 1000
#define SCREENY 1000

#define XBLOCKSIZE 16;
#define YBLOCKSIZE 16;

//float POPULATION=0.3125; //Chance, that the Random Starting Population generator decides to create a new individual
float POPULATION=0.062125; //Chance, that the Random Starting Population generator decides to create a new individual

bool g_pause = false;
bool g_singleStep = false;
bool g_gpu = true;

int GetIndividual(int x, int y, int* Array)
{
	return (Array[x+(SCREENX*y)]);
}

void SetIndividual(int x, int y, int val, int* Array)
{
	Array[x+(SCREENX*y)]=val;
}

int Neighbors(int x, int y, int* Array)
{
	int i, k, anz=0;

	for (i=-1;i<=1;i++)
		for (k=-1;k<=1;k++)
		{
			if (!((i==0)&&(k==0)) && (x+i<SCREENX) && (y+k<SCREENY) && (x+i>0) && (y+k>0))
			{
				if (GetIndividual(x+i, y+k, Array)>0)
					anz++;
			}
		}
		return anz;
}

void SpawnPopulation(float frequenzy, int* Array)
{
	int random, x,y;
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

void NextGeneration()
{
	int x, y, n, a;
	for (x=1;x<SCREENX;x++)
		for (y=1;y<SCREENY;y++)
		{
			n=Neighbors(x,y,H_a);
			a=GetIndividual(x,y,H_a);

			if (a>0)
			{
				if ((n>3) || (n<2))
					SetIndividual(x,y,0, H_b);
				else
					SetIndividual(x,y,a==255?255:a+1, H_b);
			}
			else if (GetIndividual(x,y,H_a)==0)
			{
				if (n==3)
					SetIndividual(x,y,1, H_b);
				else
					SetIndividual(x,y,0, H_b);  
			}
		}

		memcpy(H_a, H_b, SCREENX*SCREENY*sizeof(int));

}

__device__ int Dev_GetIndividual(int x,int y,int* Array)
{
	return (Array[x+(SCREENX*y)]);
}

__device__ void Dev_SetIndividual(int x, int y, int val, int* Array)
{
	Array[x+(SCREENX*y)]=val;
}

__device__ int Dev_Neighbors(int x, int y, int* Array)
{
	int i, k, anz=0;

	for (i=-1;i<=1;i++)
		for (k=-1;k<=1;k++)
		{
			if (!((i==0)&&(k==0)) && (x+i<SCREENX) && (y+k<SCREENY) && (x+i>0) && (y+k>0))
			{
				if (Dev_GetIndividual(x+i, y+k, Array)>0)
					anz++;
			}
		}
		return anz;
}

__global__ void NextGen(int* D_a, int* D_b)
{
	int a, n;

	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;


	if ((x<SCREENX)&&(y<SCREENY)&&(x>0)&&(y>0))
	{
		n=Dev_Neighbors(x,y,D_a);
		a=Dev_GetIndividual(x,y,D_a);

		if (a>0)
		{
			if ((n>3) || (n<2))
				Dev_SetIndividual(x,y,0, D_b);
			else
				Dev_SetIndividual(x,y,a==255?255:a+1, D_b);
		}
		else if (a==0)
		{
			if (n==3)
				Dev_SetIndividual(x,y,1, D_b);
			else
				Dev_SetIndividual(x,y,0, D_b);  
		}

	}

}

void CUDA_NextGeneration()
{

	int gridx=SCREENX/XBLOCKSIZE;
	int gridy=SCREENY/YBLOCKSIZE;
	int blockx=XBLOCKSIZE;
	int blocky=YBLOCKSIZE;

	cudaMemcpy(D_a, H_a, SCREENX*SCREENY*sizeof(int), cudaMemcpyHostToDevice);

	NextGen<<<dim3(gridx,gridy), dim3(blockx,blocky)>>>(D_a, D_b);
	printf("chil\n\n");
	cudaMemcpy(H_a, D_b, SCREENX*SCREENY*sizeof(int), cudaMemcpyDeviceToHost);
}

int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 2;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int timer = 0;

/*
void computeFPS()
{
    frameCount++;
    fpsCount++;

	if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetTimerValue(timer) / 1000.f);
        sprintf(fps, "The Game of Life, by Snowy: %3.1f fps %d generations", ifps, frameCount);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        //if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

        //cutilCheckError(cutResetTimer(timer));  

        //AutoQATest();
    }
 }
*/

void display()
{
	int x, y, a;    

	//cutilCheckError(cutStartTimer(timer));  

	glMatrixMode(GL_PROJECTION);	/* specifies the current matrix */
	glLoadIdentity();			/* Sets the currant matrix to identity */
	gluOrtho2D(0,SCREENX,0,SCREENY);	/* Sets the clipping rectangle extends */

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glEnable(GL_BLEND); //enable the blending
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);



	glColor3f(1,1,1);
	glPointSize (1);
	glBegin(GL_POINTS);
	for (x=1;x<SCREENX;x++)
	{
		for (y=1;y<SCREENY;y++)
		{
			a=GetIndividual(x,y, H_a);
			if (a>0)
			{
				glColor3f((float)a/255.,1.-((float)a/255.),0);   
				glVertex3f(x,y,0);
			}
		}    
	}
		
	glEnd();


	glutSwapBuffers();
	glutPostRedisplay();
	glFlush ();
	if (!g_pause || g_singleStep)
	{
		if (g_gpu)
		{
			printf("HIIIIIIII\n");
			CUDA_NextGeneration();
			}	
		else
			NextGeneration();

		g_singleStep = false;
	}
	//glutReportErrors ();
	//cutilCheckError(cutStopTimer(timer)); 
	//computeFPS();
}

void keyboard(unsigned char key, int x, int y)
{
	if (key==27)
	{  
		free(H_a);
		free(H_b);
		cudaFree(D_a);
		cudaFree(D_b); 
		exit(666);
	}
	
	if (key=='n')
	{   
		SpawnPopulation(POPULATION, H_a);
	}
	
	if (key==' ')
	{
		g_pause = !g_pause;
	}
	
	if (key=='.')
	{
		g_pause = true;
		g_singleStep = true;
	}

	if (key=='g')
	{
		g_gpu = !g_gpu;
	}
	

	display();
}

int main(int argc, char **argv)
{
	int idev, deviceCount;

	cudaDeviceProp deviceProp;
	char *device = NULL;

	if(0)//cutGetCmdLineArgumentstr(argc, (const char**)argv, "device", &device))
	{
		cudaGetDeviceCount(&deviceCount);
		idev = atoi(device);
		if(idev >= deviceCount || idev < 0)
		{
			fprintf(stderr, "Invalid device number %d, using default device 0.\n",
				idev);
			idev = 0;
		}
	}
	else
	{
		idev = 0;
	}

	//cutilSafeCall(cudaSetDevice(idev));
	cudaGetDeviceProperties(&deviceProp, idev);

	//	cutilCheckError( cutCreateTimer( &timer));

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA | GLUT_DEPTH);
	glutInitWindowSize(SCREENX, SCREENY);
	glutCreateWindow("The Game of Life, by Snowy");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);

	glClearColor(0, 0, 0, 1.0);

	H_a=(int*)malloc(SCREENX*SCREENY*sizeof(int));
	H_b=(int*)malloc(SCREENX*SCREENY*sizeof(int));
	cudaMalloc( (void**)&D_a, SCREENX*SCREENY*sizeof(int));
	cudaMalloc( (void**)&D_b, SCREENX*SCREENY*sizeof(int));

	SpawnPopulation(POPULATION, H_a);

	glutMainLoop();

	//	cutilCheckError( cutDeleteTimer( timer));
}







