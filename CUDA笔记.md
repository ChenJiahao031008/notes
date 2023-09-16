## CUDA学习笔记

[TOC]

### 一、CUDA基础知识

1. CUDA中的内存模型：

   + CUDA Core：也就是Stream Processor(SP)，是GPU最基本的处理单元。具体指令和任务都是在SP上处理的，GPU并行计算也就是很多SP同时处理。一个SP可以执行一个thread，但是实际上并不是所有的thread能够在同一时刻执行。每个线程处理器（SP）都用自己的registers（寄存器），每个SP都有自己的local memory（局部内存），register和local memory只能被线程自己访问；

   + SM：是Stream Multiprocessor，SM包含SP和一些其他资源，一个SM可以包含多个 SP。SM可以看做GPU的核心。GPU中每个SM都设计成支持数以百计的线程并行执行，并且每人GPU都包含了很多的SM，所以GPU支持成百上干的线程并行执行。每个多核处理器（SM）内都有自己的shared memory（共享内存），shared memory 可以被线程块内所有线程访问；

   + GPU设备（Device）拥有多个SM，共有一块global memory（全局内存），不同线程块的线程都可使用；

   + 图例：

     <img src="CUDA%E7%AC%94%E8%AE%B0.assets/image-20230906175231598.png" alt="image-20230906175231598" style="zoom: 67%;" />

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

5. CUDA中的共享内存

   + 共享内存的特点

     1. 读取速度等同于缓存，在很多显卡上，缓存和共享内存使用的是同一块硬
        件，并且可以配置大小
     2. 共享内存属于线程块，可以被一个线程块内的所有线程访问
     3. 共享内存的两种申请空间方式，静态申请和动态申请
     4. 共享内存的大小只有几十K，过度使用共享内存会降低程序的并行性

   + 共享内存的使用：

     1. 采用\__shared__关键字
     2. 将每个线程从全局索引位置读取元素，将它存储到共享内存之中。
     3. 注意数据存在着交叉，应该将边界上的数据拷贝进来。
     4. 块内线程同步： \_\_syncthreads()。\__syncthreads()是cuda的内建函数，用于块内线程通信。可以到达__syncthreads()的线程同步，而不是等待块内所有其他线程再同步。

   + 静态内存分配：

     1. 共享内存大小明确

        ```c++
        __global__  void staticReverse(int *d, int n) {
            __shared__ int s[64]; // 64为共享内存大小
            int t = threadIdx.x;
            int tr = n - t - 1;
            s[t] = d[t];
            __syncthreads();
            d[t] = s[tr];
        }
        
        // 1:线程组织方式，n:线程块组织方式
        // 和一般核函数没什么不同
        staticReverse<<<l, n>>>(d_d, n);
        ```

     2. 共享内存大小不明确

        ```c++
        __global__ void dynamicReverse(int *d, int n) { 
            extern __shared__ int s[]; 
            int t = threadIdx.x; 
            int tr = n - t - 1; 
            s[t] = d[t]; 
            __syncthreads(); 
            d[t] = s[tr]; 
        }
        // 第三个大小为共享内存大小
        dynamicReverse<<<1, n, 64 * sizeof(int)>>>(d_d, n);
        ```

### 二、CUDA中的矩阵乘法

1. 矩阵乘法：

   + 一个线程负责计算C中的一个元素

   + 线程矩阵中行列的计算：可画图理解

      ```c++
      i = blockIdx.y * blockDim.y + threadIdx.y // Row i of matrix C
      j = blockIdx.x * blockDim.x + threadIdx.x // Column j of matrix C
      ```

      <img src="CUDA%E7%AC%94%E8%AE%B0.assets/image-20230906185251623.png" alt="image-20230906185251623" style="zoom:50%;" />

   + 使用共享内存复用全局内存数据：将每个元素加载到共享内存中并让多个线程使用
      本地版本以减少内存带宽

   + 平铺（分块）矩阵乘法：理论加速比为Block_Size 

      + 文字叙述：

        1. 第一个平铺矩阵元素：
           $$
           \text{M[Row][tx],tx}\in(0, \text{BlockSize - 1}) \\
           \text{N[ty][Col],ty}\in(0, \text{BlockSize - 1})
           $$

        2. 第i个平铺矩阵元素：
           $$
           \text{M[Row][i * BlockSize + tx]}\in(0, \text{BlockSize - 1}) \\
           \text{N[i * BlockSize + ty][Col]}\in(0, \text{BlockSize - 1})
           $$

      + 示意图：

        <img src="CUDA%E7%AC%94%E8%AE%B0.assets/image-20230908002116406.png" alt="image-20230908002116406" style="zoom:67%;" />

      + 代码叙述：

        ```c++
        __global__ void matrixMultiplyShared(float *A, float *B, float *C,
        	int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
        {
        	__shared__ float sharedM[BLOCK_SIZE][BLOCK_SIZE];
        	__shared__ float sharedN[BLOCK_SIZE][BLOCK_SIZE];
        
        	int bx = blockIdx.x;
        	int by = blockIdx.y;
        	int tx = threadIdx.x;
        	int ty = threadIdx.y;
        	int row = by * BLOCK_SIZE + ty;
        	int col = bx * BLOCK_SIZE + tx;
        
        
        	float Csub = 0.0;
        
        	for (int i = 0; i < (int)(ceil((float)numAColumns / BLOCK_SIZE)); i++)
        	{
        		if (i * BLOCK_SIZE + tx < numAColumns && row < numARows)
        			sharedM[ty][tx] = A[row * numAColumns + i * BLOCK_SIZE + tx];
        		else
        			sharedM[ty][tx] = 0.0;
        
        		if (i * BLOCK_SIZE + ty < numBRows && col < numBColumns)
        			sharedN[ty][tx] = B[(i * BLOCK_SIZE + ty) * numBColumns + col];
        		else
        			sharedN[ty][tx] = 0.0;
        		__syncthreads();
        
        
        		for (int j = 0; j < BLOCK_SIZE; j++)
        			Csub += sharedM[ty][j] * sharedN[j][tx];
        		__syncthreads();
        	}
        
        
        	if (row < numCRows && col < numCColumns)
        		C[row * numCColumns + col] = Csub;
        
        }
        ```

### 三、CUDA 的Stream和Event

1. CUDA Stream

   + CUDA stream是GPU上task 的执行队列，所有CUDA操作（kernel，内存拷贝等）都是在stream上执行的。有点类似于CPU中的多线程，以实现数据和算法处理分离，加速计算。

   + CUDA stream有两种：

     + 默认情况下为隐式的NULL：所有的CUDA操作默认运行在隐式流里。隐式流里的GPU task和CPU端计算是同步的。

     +  另一个方法是显式申请的流。显式流里的GPU task和CPU端计算是异步的，不同显式流内的GPU task执行也是异步的。

     + 图示：

       ![image-20230910113526122](CUDA%E7%AC%94%E8%AE%B0.assets/image-20230910113526122.png)

       1. H2D 和 D2H 为什么没有重叠？

          它们已经在不同stream上了。因为CPU和GPU的数据传输是经过PCIe总线的，PCIe上的操作是顺序的。

       2. 默认流表现：

          详见PPT。

   + CUDA Stream的API：

     ```c++
     // 定义
     cudastream_t stream;
     // 创建
     cudaStreamCreate(&stream);
     // 数据传输
     cudaMemcpyAsync(dst, src, size, type, stream);
     // kernel在流中执行
     kernel_name<<<grid, block, sharedMemSize, stream>>>(argument list);
     // 同步和查询
     cudaError_t cudaStreamSynchronize(cudaStream_t stream); // 同步一个流
     cudaError_t cudaDeviceSynchronize(); // 同步该设备上的所有流
     cudaError_t cudaStreamQuery(cudaStream_t stream);
     // 销毁
     cudaError t cudaStreamDestroy(cudastream t stream);
     ```

   + CUDA Stream的Demo应用：

     ```c++
     // 创建两个流
     cudaStream_t stream[2];
     // 定义
     for(inti=0;i<2,++i)
     	cudaStreamCreate(&stream[i]);
     float* hostPtr;
     cudaMallocHost(&hostPtr, 2 * size);
     // ...
     // 两个流，每个流有三个命令
     for(int i=0; i<2 ; ++i){
         //从主机内存复制数据到设备内存
         cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
         //执行Kernel处理
         MyKernel<<<grid, block, 0, stream[i]>>>(outputDevPtr + i * size, inputDevPtr + i * size, size);
         /从设备内存到主机内存
         cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
     }
     
     //同步流
     for (int i=0; i<2; i++)
     	cudaStreamSynchronize(stream[i]);
     //销毁流
     for (int i=0; i<2; ++i)
     	cudastreamDestroy(stream[i]);
     ```

2. CUDA Event

   + CUDA Event API：

     ```c++
     // 定义
     cudaEvent_t event
     // 创建
     cudaError_t cudaEventCreate(cudaEvent_t* event);
     // 插入流中
     cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
     // 销毁
     cudaError_t cudaEventDestroy(cudaEvent_t event);
     // 同步和查询
     cudaError_t cudaEventSynchronize(cudaEvent_t event);
     cudaError_t cudaEventQuery(cudaEvent_t event);
     // 进阶同步函数
     cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event);
     ```

   + CUDA Event Demo：计时实现

     ```c++
     float time_elapsed=0;
     // 开始和结束两个事件
     cudaEvent_t start,stop;
     //创建Event
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     //记录当前时间
     cudaEventRecord(start, 0);
     // 调用核函数
     mul<<<blocks, threads, 0, 0>>>(dev_a, NUM);
     //记录当前时间
     cudaEventRecord(stop, 0);
     // waits for an event to complete.
     cudaEventSynchronize(start);
     cudaEventSynchronize(stop);
     // waits foranevent to complete.Record之前的任务
     cudaEventElapsedTime(&time_elapsed, start, stop);
     //计算时间差
     cudaEventDestroy(start);
     //destory the event
     cudaEventDestroy(stop);
     printf（"执行时间：%f（ms）\n"，time_elapsed）；
     ```

3. CUDA 显示同步操作：

   + device synchronize：影响范围最大，涉及所有的stream和cpu；

     <img src="CUDA%E7%AC%94%E8%AE%B0.assets/image-20230914222822447.png" alt="image-20230914222822447" style="zoom:50%;" />

   + stream synchronize： 影响单个流和CPU；

     <img src="CUDA%E7%AC%94%E8%AE%B0.assets/image-20230914223557903.png" alt="image-20230914223557903" style="zoom:50%;" />

   + event synchronize： 影响单个流和CPU；

     <img src="CUDA%E7%AC%94%E8%AE%B0.assets/image-20230914223647666.png" alt="image-20230914223647666" style="zoom:50%;" />

   + cudaStreamWaitEvent：绕过CPU进行同步，函数会指定该stream等待特定的event，该event可以关联到相同或者不同的stream；

     <img src="CUDA%E7%AC%94%E8%AE%B0.assets/image-20230914223918145.png" alt="image-20230914223918145" style="zoom: 33%;" />

4. NVIDIA Visual Profiler（NVVP）: GPU性能分析工具

### 四、cublas Library 

1. 介绍：
   + Cublas Library 就是在NVIDIA CUDA中实现**Blas基本线性代数子程序**。它允许用户访问NVIDIA中GPU（图形处理单元）的计算资源，但不能同时对多个GPU进行自动并行访问。
   + Cublas 实现了三类函数向量标量、向量矩阵、矩阵矩阵，并通过头文件` include "cublas_v2.h“`引用。
   + Cublas library 同时还提供了从GPU中书写和检索数据的功能。
2. 学习网站：https://docs.nvidia.com/cuda/cublas/index.html
3. 数据布局：对于现有的具有最大兼容性的Fortran环境，Cublas library**使用列主序存储**和**1-based indexing（以1开始索引）**。这和我们编程习惯有很大的差异。

### 五、cuDNN Library 

