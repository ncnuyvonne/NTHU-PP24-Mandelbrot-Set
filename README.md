# NTHU-PP24-Mandelbrot-Set
* name：陳郁芳
* student ID：0021-111321007
## Implementation
### Load Balancing (Partition)
使每個rank計算的資料量相差不超過1行。
#### Pthread
```c=
pthread_t threads[ncpus];
int ranges[ncpus][2];
int chunksize = height / ncpus;
int rest = height%ncpus;
for(int x = 0; x < ncpus; x++) {
    ranges[x][0] = x * chunksize + ((x < rest) ? x : rest);
    ranges[x][1] = ranges[x][0] + ((x < rest) ? chunksize+1 : chunksize);
    //...
}
```
#### Hybrid
在hw2b中還需要透過`MPI_Gatherv`將分布在各個rank的part_image收集到rank 0，由rank 0產生png檔，因此這裡也需要重新計算偏移量。
```c=
for (int i = 0; i < size; i++) {
        int local_chunk = (i < rest) ? (chunksize + 1) : chunksize; 
        recvcounts[i] = local_chunk * width;
        displs[i] = (i == 0) ? 0 : (displs[i - 1] + recvcounts[i - 1]);
}
```
### Hybrid Parallelism
我依序總共寫了四個版本：openmp、mpi、hybrid(omp+mpi)、Vectorization，其中使用hybrid(omp+mpi)、Vectorization會讓效能得到更顯著的提升。
hybrid的版本即完成mpi後再加入openmp。
Vectorization主要是對Mandelbrot Set的演算法進行SIMD的加速，但是openmp的程式放在演算法裡會出現錯誤，所以整體會減少openmp的使用。
### Vectorization
我使用的Vector Instruction Set是avx512，它的flag是`-mavx512f`。
* `#include <immintrin.h>`：提供avx512 SIMD指令的data type和function。
* `__m512d`：512bits vector的data type，可以放8個double。
* `_mm512_set_pd`：設置vector的資料，可以放入8個不同的double。
* `_mm512_set1_pd`：設置vector的資料，可以放入8個相同的double
* `_mm512_setzero_pd`：初始化vector為0。
* `_mm512_mul_pd`：vector乘法。
* `_mm512_add_pd`：vector加法。
* `_mm512_sub_pd`：vector減法。
* `_mm512_cmp_pd_mask`：比較vector，回傳1或0(True or False)。
    * `__mmask8`：存放比較結果的data type。
    * `_CMP_LT_OQ`：小於。
* `_mm512_fmadd_pd`：Fused Multiply-Add, AxB+C。
## Experiment & Analysis
### Methodology
* System Spec：qct cluster ([Lab2簡報](https://docs.google.com/presentation/d/18pM-bdGmtEzCf6_Ztc14JJrRQd6w8a0iQ_yL9kuwrgE/edit#slide=id.g2f8a0bdfafa_0_5))
* Performance Metrics: 我使用`time`來計算各自的時間。
### Plots: Speedup Factor & Profile
#### Experimental Method:
 * Test Case Description: 我使用的是strict34.in這筆測資，因為這是我跑最慢的測資。
 * Parallel Configurations:
     1. 在Pthread中，我用單process，thread數量對時間的影響。
     2. 在Hybrid中，我用多process，load balancing的情況。
#### Analysis of Results: 
##### Time Profile
![1](https://i.imgur.com/l7t4pYG.png)
*圖1, 單process，thread數量對時間的影響* 
![2](https://i.imgur.com/N1AlW6v.png)
*圖2, process數量與時間的比較*
##### Strong Scalability
![3](https://i.imgur.com/N6JZ9TO.png)
*圖3, 單process，thread數量對時間的影響*
##### Load Balancing
![4](https://i.imgur.com/ZrDyGYf.png)
*圖4, 多process，load balancing的情況(8 process)*
#### Optimization Strategies
其實我很猶豫要不要放這張(圖2)，但它確實是測出來的數據。
如上所說，因為我在coding過程中，發現加入太多openmp程式會導致make出現錯誤，所以在最後的版本，每個process中除了SIMD以外使用openmp平行的不多，我認為是這個原因導致在Hybrid中沒有良好的Scalability。
如果想要在Hybrid得到良好的Scalability，應該要對每個process分發的thread有更細節的處理，想辦法讓Vectorization和openmp的結合有更好的效能。
## Experiences / Conclusion
因為odd-even sort花了我不少時間，所以Mandelbrot Set沒能有更多時間進行profile(deadline隔天是期中考，但我還是弄到晚上QAQ)，但我覺得最酷的是Vector Instruction Set，這是以前從來沒有玩過的，而且用的好，加速的效果非常驚人，但是要學好線性代數。
另外就是因為臨近deadline我來不及問，這台qct server要怎麼把nsys檔案複製出來？我用scp是失敗的，應該是因為不在一條link上，但我不知道有什麼其他方法取出來(這就是為什麼報告沒有寫nsys的profile結果)。
