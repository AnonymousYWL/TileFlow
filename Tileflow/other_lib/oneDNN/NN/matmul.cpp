/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example matmul.cpp
/// > Annotated version: @ref matmul_example_cpp
///
/// @page matmul_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [MatMul](@ref dev_guide_matmul) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes with fused post-ops.
///
/// @page matmul_example_cpp Matmul Primitive Example
/// @copydetails matmul_example_cpp_short
///
/// @include matmul.cpp
/*
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <random> 

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

#define NUM 64

using tag = memory::format_tag;
using dt = memory::data_type;

static double gtod_ref_time_sec = 0.0;

double dclock()
{
    double the_time, norm_sec;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    if (gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (double)tv.tv_sec;

    norm_sec = (double)tv.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + tv.tv_usec * 1.0e-6;

    return the_time;
}


void matmul_example(dnnl::engine::kind engine_kind)
{

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);


    long Ns[] = { 28672, 8192, 36864, 4608, 25600, 5120, 16384, 6144, 13824, 5120, 14336, 4096, 11008, 7168, 4096, 10240, 16545, 11243};
    long Ks[] = { 8192, 28672, 4608, 36864, 5120, 25600, 6144, 16384, 5120, 13824, 4096, 14336, 4096, 4096, 7168, 192, 128, 96};    

    FILE *fp;
    if ((fp = fopen("./onednn_NN_FP32_6430_table_UMA.txt", "w")) == NULL)
    {
        puts("Fail to open file!");
        exit(0);
    }

    for (int t = 2; t < 3; t+=1)
    {
        for (int k =16; k < 257; k+=16)
        {
            // Tensor dimensions.
            const memory::dim N = Ns[t], M = k, K =Ks[t];
            fprintf(fp, "%d %d %d", M,N,K);
            // Source (src), weights, and destination (dst) tensors dimensions.
            memory::dims src_dims = {M, K};
            memory::dims weights_dims = {K, N};
            memory::dims dst_dims = {M, N};

	    for(int j=0; j < 5; j++)
	    {
            // Allocate buffers.
            std::vector<uint16_t> src_data(product(src_dims));
            std::vector<uint16_t> weights_data(product(weights_dims));
            std::vector<float> dst_data(product(dst_dims));
            

            // Randomly initialize src and weights between -1 and 1.
            #pragma omp parallel for
            for (size_t i = 0; i < src_data.size(); i++)
            {
                thread_local std::mt19937 gen(std::random_device{}());
                thread_local std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
                src_data[i] = dis(gen) / RAND_MAX * 2.0f - 1.0f;
            }
            
            #pragma omp parallel for
            for (size_t i = 0; i < weights_data.size(); i++)
            {
                thread_local std::mt19937 gen(std::random_device{}());
                thread_local std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
                weights_data[i] = dis(gen) / RAND_MAX * 2.0f - 1.0f;
            }

            // Create memory descriptors and memory objects for src, weights, and dst.
            auto src_md = memory::desc(src_dims, dt::bf16, tag::ab);
            auto weights_md = memory::desc(weights_dims, dt::bf16, tag::ab);
            auto dst_md = memory::desc(dst_dims, dt::f32, tag::ab);

            auto src_mem = memory(src_md, engine);
            auto weights_mem = memory(weights_md, engine);
            auto dst_mem = memory(dst_md, engine);

            // Write data to memory object's handles.
            write_to_dnnl_memory(src_data.data(), src_mem);
            write_to_dnnl_memory(weights_data.data(), weights_mem);

            // Create primitive descriptor.
            auto matmul_pd = matmul::primitive_desc(engine, src_md, weights_md, dst_md);

            // Create the primitive.
            auto matmul_prim = matmul(matmul_pd);

            // Primitive arguments.
            std::unordered_map<int, memory> matmul_args;
            matmul_args.insert({DNNL_ARG_SRC, src_mem});
            matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
            matmul_args.insert({DNNL_ARG_DST, dst_mem});

            // Number of iterations for performance testing
            const int num_iterations = 100;

                // Primitive execution: matrix multiplication.
                matmul_prim.execute(engine_stream, matmul_args);

                // Wait for the computation to finalize.
                engine_stream.wait();

                // Start time measurement
                double start = 0, cost =0;
		start = dclock();
                for (int i = 0; i < num_iterations; ++i)
                {
                    // Primitive execution: matrix multiplication.
                    matmul_prim.execute(engine_stream, matmul_args);

                    // Wait for the computation to finalize.
                    engine_stream.wait();
                }
		cost =  (dclock() - start)/num_iterations;

                // End time measurement
                //double cost = (dclock() - start) / num_iterations;
                double ops = (double)M * N * K * 1.0e-12 * 2;
    		printf("oneDNN_FP32: M= %-6d N=%-6d K=%-6d Tflops = %-8.3lf \n", M, N, K, ops / cost);
                fprintf(fp, " %.3f", M, N, K, ops/cost);
	    }
	    fprintf(fp, "\n");
            printf("------------------------------------------------\n");
        }
        printf("\n\n");
    }
    fclose(fp);
}*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <random>
#include <omp.h>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

#define NUM 64


using tag = memory::format_tag;
using dt = memory::data_type;

static double gtod_ref_time_sec = 0.0;

double dclock()
{
    double the_time, norm_sec;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    if (gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (double)tv.tv_sec;

    norm_sec = (double)tv.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + tv.tv_usec * 1.0e-6;

    return the_time;
}

// 将 FP32 转换为 BF16
uint16_t FP32_toBF16(float fp32_value) 
{
    // 将浮点数的二进制表示强制转换为无符号 32 位整数
    uint32_t fp32_bits;
    memcpy(&fp32_bits, &fp32_value, sizeof(fp32_bits));
    
    // 截取高 16 位作为 BF16 格式，低 16 位丢弃（精度截断）
    uint16_t bf16_bits = (uint16_t)(fp32_bits >> 16);
    
    return bf16_bits;
}

// 将 BF16 转换回 FP32
float BF16_toFP32(uint16_t bf16_value) {
    // 创建一个 32 位的无符号整数，将 BF16 的高 16 位放入 FP32 的高位
    uint32_t fp32_bits = (uint32_t)bf16_value << 16;
    
    // 将二进制表示强制转换回浮点数
    float fp32_value;
    memcpy(&fp32_value, &fp32_bits, sizeof(fp32_value));
    
    return fp32_value;
}

void matmul_example(dnnl::engine::kind engine_kind)
{
    engine engine(engine_kind, 0);
    stream engine_stream(engine);

    long Ns[] = {28672, 8192,  36864, 16384, 25600, 13824, 14336};
    long Ks[] = {8192,  28672, 4608,  6144,  5120,  5120,  4096};

    FILE *fp;
    fp = fopen("./oneDNN_NN_FP32_6430_exp2.txt", "w");

    for (int t = 0; t < 7; t++)
    {
        int BUF=1;
        if(t<3)
            BUF=10;

        for (int k = 1; k < 257; k++)
        {

            const memory::dim N = Ns[t];
            const memory::dim M = k;
            const memory::dim K = Ks[t];

            fprintf(fp, "%ld %ld %ld", M, N, K);
            
            
            for(int pc=0; pc < 5; pc++)
            {

            memory::dims src_dims = {M, K};
            memory::dims weights_dims = {K, N};
            memory::dims dst_dims = {M, N};

            /* ----------------- Allocate buffers ----------------- */

            std::vector<uint16_t> src_data(M * K);

            std::vector<std::vector<uint16_t>> weights_data(BUF);
            std::vector<std::vector<float>> dst_data(BUF);

            for (int i = 0; i < BUF; i++)
            {
                weights_data[i].resize(K * N);
                dst_data[i].resize(M * N);
            }

            /* ----------------- Random init ----------------- */

            #pragma omp parallel for
            for (size_t i = 0; i < src_data.size(); i++)
            {
                thread_local std::mt19937 gen(std::random_device{}());
                thread_local std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
                src_data[i] = FP32_toBF16(dis(gen));
            }

            for (int b = 0; b < BUF; b++)
            {
                #pragma omp parallel for
                for (size_t i = 0; i < weights_data[b].size(); i++)
                {
                    thread_local std::mt19937 gen(std::random_device{}());
                    thread_local std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
                    weights_data[b][i] = FP32_toBF16(dis(gen));;
                }
            }

            /* ----------------- memory descriptor ----------------- */

            auto src_md = memory::desc(src_dims, dt::bf16, tag::ab);
            auto weights_md = memory::desc(weights_dims, dt::bf16, tag::ab);
            auto dst_md = memory::desc(dst_dims, dt::f32, tag::ab);

            memory src_mem(src_md, engine, src_data.data());

            std::vector<memory> weights_mem(BUF);
            std::vector<memory> dst_mem(BUF);

            for (int i = 0; i < BUF; i++)
            {
                weights_mem[i] = memory(weights_md, engine, weights_data[i].data());
                dst_mem[i] = memory(dst_md, engine, dst_data[i].data());
            }

            /* ----------------- matmul primitive ----------------- */

            auto matmul_pd = matmul::primitive_desc(engine, src_md, weights_md, dst_md);
            auto matmul_prim = matmul(matmul_pd);

            const int num_iterations = 100;

            /* warmup */
            for (int i = 0; i < BUF; i++)
            {
                int id = i % BUF;

                std::unordered_map<int, memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weights_mem[id]});
                args.insert({DNNL_ARG_DST, dst_mem[id]});

                matmul_prim.execute(engine_stream, args);
            }

            engine_stream.wait();

            /* ----------------- benchmark ----------------- */

            double start = dclock();

            for (int i = 0; i < num_iterations; i++)
            {
                int id = i % BUF;

                std::unordered_map<int, memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weights_mem[id]});
                args.insert({DNNL_ARG_DST, dst_mem[id]});

                matmul_prim.execute(engine_stream, args);
            }

            engine_stream.wait();

            double cost = (dclock() - start) / num_iterations;

            double ops = (double)M * N * K * 2e-12;

            printf("oneDNN_FP32: M=%-6ld N=%-6ld K=%-6ld cost= %-6.3f ms Tflops=%-8.3lf\n",
                   M, N, K, cost * 1000,ops / cost);

            fprintf(fp, " %.3f", ops / cost);
            }
            fprintf(fp, "\n");

            printf("------------------------------------------------\n");
        }
    }

    fclose(fp);
}
int main(int argc, char **argv) {
    return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
}

