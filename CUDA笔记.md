## CUDA学习

### 一、基础知识

1. CUDA中的内存模型：

   + CUDA Core：也就是Stream Processor(SP)，是GPU最基本的处理单元。具体指令和任务都是在SP上处理的，GPU并行计算也就是很多SP同时处理。一个SP可以执行一个thread，但是实际上并不是所有的thread能够在同一时刻执行。每个线程处理器（SP）都用自己的registers（寄存器），每个SP都有自己的local memory（局部内存），register和local memory只能被线程自己访问；

   + SM：是Stream Multiprocessor，SM包含SP和一些其他资源，一个SM可以包含多个 SP。SM可以看做GPU的核心。GPU中每个SM都设计成支持数以百计的线程并行执行，并且每人GPU都包含了很多的SM，所以GPU支持成百上干的线程并行执行。每个多核处理器（SM）内都有自己的shared memory（共享内存），shared memory 可以被线程块内所有线程访问；

   + GPU设备（Device）拥有多个SM，共有一块global memory（全局内存），不同线程块的线程都可使用；

   + 从硬件角度：

     + 线程块是一堆线程的集合，在线程块中的线程被布局成一维，每 32个连续线程组成一个线程束（32这个数字是硬件端已经设计好的）。所以线程在软件端可以被布局成一维，二维和三维。通过计算线程ID 映射到硬件上的物理单元上。
     + SM采用的单指令多线程架构，warp(线程束)本质上是线程在GPU上运行的最小单元，一个warp包含32个并行thread，所以block所含的thread的大小一般要设置为32的倍数。
     + GPU在线程调度的时候，一个warp需要占用一个SM运行，多个warps会由SM的硬件warp scheduler负责调度，每次选择一个线程束分配到 SM 上。1个 SM 中的 Warp Scheduler 的个数和SP 的个数决定了到底 1个 SM 可以同时运行几个warp。

   + 软硬件对比：

     | 软件概念               | 硬件概念         |
     | ---------------------- | ---------------- |
     | 线程（thread）         | 线程处理器（SP） |
     | 线程块（thread block） | 多核处理器（SM） |
     | 线程块组合体（grid）   | 设备端（device） |

     + 一个 kernel 其实由一个grid来执行，一个 kernel 一次只能在一个GPU上执行；

   + 线程束warp在软硬件端的执行过程：

     + 一人网格被启动（网格被启动，等价于一人内核被启动，每人内核对应于自己的网格），网格中包含线程块；
     + 线程块被分配到某一个 SM 上；
     + SM上的线程块将分为多个线程束，每个线程束一般是 32个线程；
     + 在一个线程束中，所有线程按照单指令多线程SIMT的方式执行。

2. 核函数调用

   + 在GPU上执行的函数。
   + 一般通过标识符\__global__修饰。
   + 调用通过<<<参数1,参数2>>>，用于说明内核函数中的线程数量，以及线程是如何组织的。
   + 以网格（Grid）的形式组织，每个线程格由若干个线程块（block）组成，而每个线程块又由若干个线程
     （thread）组成。
   + 调用时必须声明内核函数的执行参数。
   + 在编程时，必须先为kernel函数中用到的数组或变量分配好足够的空间，再调用kernel函数，否则在GPU计算时会发生错误。

3. CUDA简单样例：

   ```c++
   __global__
   void VecAddKernel(float* A_d, float* B_d, float* C_d, int n)
   {
       int i = threadIdx.x + blockDim.x * blockIdx.x;
       if(i < n) C_d[i] = A_d[i] + B_d[i];
   }
   
   void VecAddMain(float* A, float* B, float* C, int n)
   {
   	int size = n ＊ sizeof(float);
   	float* A_d, *B_d, *C_d;
       // Transfer A and B to device memory
   	cudaMalloc((void **) &A_d, size);
       // void *dst, const void *src, size_t count, cudaMemcpyKind kind
   	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
   	cudaMalloc((void **) &B_d, size);
   	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
   	// Allocate device memory for
   	cudaMalloc((void **) &C_d, size);
   	// Kernel invocation code - to be shown later
       // Run ceil(n/256) blocks of 256 threads each
       // blockPerGrid, threadPerBlock
       VecAddKernel<<< ceil(n/256), 256 >>>(A_d, B_d, C_d, n);
   	// Transfer C from device to host
   	cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
   	// Free device memory for A, B, C
       cudaFree(A_d); cudaFree(B_d); cudaFree (C_d);
   }
   
   ```

   编译指令：

   ```makefile
   /usr/local/cuda/bin/nvcc main_gpu.cu -o VectorSumGPU
   ```

   其中：

   |          编程标识符号          | 工作地点 | 被调用地点 |
   | :----------------------------: | :------: | :--------: |
   | \__device__ float DeviceFunc() |  device  |   device   |
   | \__global__ void KernelFunc()  |  device  |    host    |
   |   \__host__ float HostFunc()   |   host   |    host    |

4. 线程ID的计算：

   + Thread，block，grid是CUDA编程上的概念，为了方便程序员软件设计，组织线程。CUDA的软件架构由网格（Grid）、线程块（Block）和线程（Thread）组成，相当于把GPU上的计算单元分为若干（2~3）个网格，每个网格内包含若干（65535）个线程块，每人线程块包含若干(512)个线程。

   + 一个Grid可以包含多人Blocks，Blocks的组织方式可以是一维的，二维或者三维的。block包含多个Threads，这些Threads的组织方式也可以是一维，二维或者三维的。CUDA中每一个线程都有一个唯一的标识ID一Threadldx，这个ID随着Grid和Block的划分方式的不同而变化。

   + 计算ID过程：

     ```c++
     // 1. grid划分成1维，block划分为1维
     int threadId = blockIdx.x * blockDim.x + threadIdx.x;
     
     // 2. grid划分成1维，block划分2维
     int threadId = blockIdx.x * blockDim.x * blockDim.y 
         		 + threadIdx.y * blockDim.x 
         		 + threadIdx.x;
     
     // 3. grid划分成1维，block划分为3维
     int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z 
         		 + threadIdx.z * blockDim.y * blockDim.x
           		 + threadIdx.y * blockDim.x 
         		 + threadIdx.x;
     
     // 4. grid划分成2维，block划分为1维
     int blockId = blockIdx.y * gridDim.x + blockIdx.x;
     int threadId = blockId * blockDim.x + threadIdx.x;
     
     // 5. grid划分成2维，block划分为2维
     int blockId = blockIdx.y * gridDim.x + blockIdx.x;
     int threadId = blockId * blockDim.x * blockDim.y
        			 + threadIdx.y * blockDim.x
         		 + threadIdx.x
     
     // 6. grid划分成2维，block划分为3维
     int blockId = blockIdx.x + blockIdx.y * gridDim.x;
     int threadId = blockId * blockDim.x * blockDim.y * blockDim.z
     			 + threadIdx.z * blockDim.x * blockDim.y
     			 + threadIdx.y * blockDim.x 
         		 + threadIdx.X;
     
     // 7. grid划分成3维，block划分为1维
     int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
     int threadId = blockId * blockDim.x + threadIdx.x;
     
     // 8. grid划分成3维，block划分为2维
     int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.zj
     int threadId = blockId * blockDim.x * blockDim.y 
         		 + threadIdx.y * blockDim.x 
         		 + threadIdx.x;
     
     // 9. grid划分成3维，block划分为3维
     int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
     int threadId = blockId * blockDim.x * blockDim.y * blockDim.z
     			 + threadIdx.z * blockDim.x * blockDim.y
     			 + threadIdx.y * blockDim.x 
         		 + threadIdx.X;
     
     ```

     
