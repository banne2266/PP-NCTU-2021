#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
#define BLOCK_SIZE 16

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int dataSize = imageHeight * imageWidth * sizeof(float);

    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);

    cl_mem input_image_d = clCreateBuffer(*context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
    cl_mem filter_d = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, &status);
    cl_mem output_image_d = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);

    status = clEnqueueWriteBuffer(command_queue, input_image_d, CL_TRUE, 0, dataSize, inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, filter_d, CL_TRUE, 0, filterSize, filter, 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&filterWidth);
    status = clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&imageHeight);
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&imageWidth);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&input_image_d);
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&filter_d);
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&output_image_d);

    size_t localWorkSize[] = {BLOCK_SIZE, BLOCK_SIZE};
    size_t globalWorkSize[] = {(imageHeight+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE, (imageWidth+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE};
    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    status = clFlush(command_queue);
    clFinish(command_queue);
    
    status = clEnqueueReadBuffer(command_queue, output_image_d, CL_TRUE, 0, dataSize, outputImage, 0, NULL, NULL);
    cl_event GPUDone;
    clWaitForEvents(0, GPUDone);

    status = clFinish(command_queue);
    status = clReleaseKernel(kernel);
    status = clReleaseMemObject(input_image_d);
    status = clReleaseMemObject(filter_d);
    status = clReleaseMemObject(output_image_d);
    status = clReleaseCommandQueue(command_queue);

    return 0;
}