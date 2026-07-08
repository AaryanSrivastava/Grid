#include <Grid/GridCore.h>

NAMESPACE_BEGIN(Grid);
int      world_rank; // Use to control world rank for print guarding
int      acceleratorAbortOnGpuError=1;
uint32_t accelerator_threads=2;
uint32_t acceleratorThreads(void)       {return accelerator_threads;};
void     acceleratorThreads(uint32_t t) {accelerator_threads = t;};

#define ENV_LOCAL_RANK_PALS    "PALS_LOCAL_RANKID"
#define ENV_RANK_PALS          "PALS_RANKID"
#define ENV_LOCAL_RANK_OMPI    "OMPI_COMM_WORLD_LOCAL_RANK"
#define ENV_RANK_OMPI          "OMPI_COMM_WORLD_RANK"
#define ENV_LOCAL_RANK_SLURM   "SLURM_LOCALID"
#define ENV_RANK_SLURM         "SLURM_PROCID"
#define ENV_LOCAL_RANK_MVAPICH "MV2_COMM_WORLD_LOCAL_RANK"
#define ENV_RANK_MVAPICH       "MV2_COMM_WORLD_RANK"

#ifdef GRID_CUDA
cudaDeviceProp *gpu_props;
cudaStream_t copyStream;
cudaStream_t computeStream;
void acceleratorInit(void)
{
  int nDevices = 1;
  cudaGetDeviceCount(&nDevices);
  gpu_props = new cudaDeviceProp[nDevices];

  char * localRankStr = NULL;
  int rank = 0;
  world_rank=0; 
  if ((localRankStr = getenv(ENV_RANK_OMPI   )) != NULL) { world_rank = atoi(localRankStr);}
  if ((localRankStr = getenv(ENV_RANK_MVAPICH)) != NULL) { world_rank = atoi(localRankStr);}
  if ((localRankStr = getenv(ENV_RANK_SLURM  )) != NULL) {
    int slurm_rank = atoi(localRankStr);
    if (world_rank != 0 && world_rank != slurm_rank) {
      std::cout << "Both SLURM and OMPI ranks detected and differ - " << slurm_rank << " != " << world_rank << "! Preferring the OMPI rank." << std::endl;
    } else {
      world_rank = slurm_rank;
    }
  }
  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv(ENV_LOCAL_RANK_OMPI)) != NULL) {
    if (!world_rank)
      std::cout << "OPENMPI detected" << std::endl;
    rank = atoi(localRankStr);		
  } else if ((localRankStr = getenv(ENV_LOCAL_RANK_MVAPICH)) != NULL) {
    if (!world_rank)
      std::cout << "MVAPICH detected" << std::endl;
    rank = atoi(localRankStr);		
  } else if ((localRankStr = getenv(ENV_LOCAL_RANK_SLURM)) != NULL) {
    if (!world_rank)
      std::cout << "SLURM detected" << std::endl;
    rank = atoi(localRankStr);		
  } else { 
    if (!world_rank)
      std::cout << "MPI version is unknown - bad things may happen" << std::endl;
  }

  size_t totalDeviceMem=0;
  for (int i = 0; i < nDevices; i++) {

#define GPU_PROP_FMT(__name__,__value__)     std::cout << "AcceleratorCudaInit[" << rank << "]:   " #__name__ ": " << __value__ << std::endl;
#define GPU_PROP(__name__)             GPU_PROP_FMT(__name__, prop.__name__);
    cudaGetDeviceProperties(&gpu_props[i], i);
    cudaDeviceProp prop; 
    prop = gpu_props[i];
    totalDeviceMem = prop.totalGlobalMem;
    if ( world_rank == 0) {
      if ( i==rank ) {
	std::cout << "AcceleratorCudaInit[" << rank << "]: ========================" << std::endl;
	GPU_PROP_FMT(Device Number, i);
	std::cout << "AcceleratorCudaInit[" << rank << "]: ========================" << std::endl;
	GPU_PROP_FMT(Device identifier, prop.name);

	GPU_PROP(totalGlobalMem);
	GPU_PROP(managedMemory);
	GPU_PROP(isMultiGpuBoard);
	GPU_PROP(warpSize);
	GPU_PROP(pciBusID);
	GPU_PROP(pciDeviceID);
	std::cout << "AcceleratorCudaInit[" << rank << "]: maxGridSize (" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << ")" << std::endl;
      }
      //      GPU_PROP(unifiedAddressing);
      //      GPU_PROP(l2CacheSize);
      //      GPU_PROP(singleToDoublePrecisionPerfRatio);
    }
  }

  MemoryManager::DeviceMaxBytes = (8*totalDeviceMem)/10; // Assume 80% ours
#undef GPU_PROP_FMT    
#undef GPU_PROP

#ifdef GRID_DEFAULT_GPU
  int device = 0;
  // IBM Jsrun makes cuda Device numbering screwy and not match rank
  if ( world_rank == 0 ) {
    std::cout << "AcceleratorCudaInit: using default device" << std::endl;
    std::cout << "AcceleratorCudaInit: assume user either uses" << std::endl;
    std::cout << "AcceleratorCudaInit: a) IBM jsrun, or " << std::endl;
    std::cout << "AcceleratorCudaInit: b) invokes through a wrapping script to set CUDA_VISIBLE_DEVICES, UCX_NET_DEVICES, and numa binding " << std::endl;
    std::cout << "AcceleratorCudaInit: Configure options --enable-setdevice=no " << std::endl;
  }
#else
  int device = rank;
  std::cout << "AcceleratorCudaInit: rank " << world_rank << " setting device to node rank " << rank << std::endl;
  std::cout << "AcceleratorCudaInit: Configure options --enable-setdevice=yes " << std::endl;
#endif

  cudaSetDevice(device);
  cudaStreamCreate(&copyStream);
  cudaStreamCreate(&computeStream);
  const int len=64;
  char busid[len];
  if( rank == world_rank ) { 
    cudaDeviceGetPCIBusId(busid, len, device);
    std::cout << "local rank " << rank << " device " << device << " bus id: " << busid << std::endl;
  }

  if ( world_rank == 0 )  std::cout << "AcceleratorCudaInit: ================================================" << std::endl;
}
#endif

#ifdef GRID_HIP
hipDeviceProp_t *gpu_props;
hipStream_t copyStream;
hipStream_t computeStream;
void acceleratorInit(void)
{
  int nDevices = 1;
  auto discard = hipGetDeviceCount(&nDevices);
  gpu_props = new hipDeviceProp_t[nDevices];

  char * localRankStr = NULL;
  int rank = 0;
  world_rank=0; 
  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv(ENV_LOCAL_RANK_OMPI)) != NULL)
  {
    rank = atoi(localRankStr);		
  }
  if ((localRankStr = getenv(ENV_LOCAL_RANK_MVAPICH)) != NULL)
  {
    rank = atoi(localRankStr);		
  }
  if ((localRankStr = getenv(ENV_RANK_OMPI   )) != NULL) { world_rank = atoi(localRankStr);}
  if ((localRankStr = getenv(ENV_RANK_MVAPICH)) != NULL) { world_rank = atoi(localRankStr);}
  if ((localRankStr = getenv(ENV_RANK_SLURM  )) != NULL) { world_rank = atoi(localRankStr);}

  if ( world_rank == 0 ) 
    printf("world_rank %d has %d devices\n",world_rank,nDevices);
  size_t totalDeviceMem=0;
  for (int i = 0; i < nDevices; i++) {

#define GPU_PROP_FMT(__name__, __value__)     std::cout << "AcceleratorHipInit:   " #__name__ ": " << __value__ << std::endl;
#define GPU_PROP(__name__)             GPU_PROP_FMT(__name__, prop.__name__);
    
    discard = hipGetDeviceProperties(&gpu_props[i], i);
    hipDeviceProp_t prop; 
    prop = gpu_props[i];
    totalDeviceMem = prop.totalGlobalMem;
    if ( world_rank == 0) {
      std::cout << "AcceleratorHipInit[" << rank << ": ========================" << std::endl;
      GPU_PROP_FMT(Device Number, i);
      std::cout << "AcceleratorHipInit[" << rank << ": ========================" << std::endl;
      GPU_PROP_FMT(Device identifier, prop.name);

      GPU_PROP(totalGlobalMem);
      //      GPU_PROP(managedMemory);
      GPU_PROP(isMultiGpuBoard);
      GPU_PROP(warpSize);
      //      GPU_PROP(unifiedAddressing);
      //      GPU_PROP(l2CacheSize);
      //      GPU_PROP(singleToDoublePrecisionPerfRatio);
    }
  }
  MemoryManager::DeviceMaxBytes = (8*totalDeviceMem)/10; // Assume 80% ours
#undef GPU_PROP_FMT    
#undef GPU_PROP

#ifdef GRID_DEFAULT_GPU
  if ( world_rank == 0 ) {
    std::cout << "AcceleratorHipInit: using default device " << std::endl;
    std::cout << "AcceleratorHipInit: assume user or srun sets ROCR_VISIBLE_DEVICES and numa binding " << std::endl;
    std::cout << "AcceleratorHipInit: Configure options --enable-setdevice=no " << std::endl;
  }
  int device = 0;
#else
  if ( world_rank == 0 ) {
    std::cout << "AcceleratorHipInit: rank " << world_rank << " setting device to node rank " << rank << std::endl;
    std::cout << "AcceleratorHipInit: Configure options --enable-setdevice=yes " << std::endl;
  }
  int device = rank;
#endif
  discard = hipSetDevice(device);
  discard = hipStreamCreate(&copyStream);
  discard = hipStreamCreate(&computeStream);
  const int len=64;
  char busid[len];
  if( rank == world_rank ) { 
    discard = hipDeviceGetPCIBusId(busid, len, device);
    std::cout  << "local rank " << rank << " device " << device << " bus id: " << busid << std::endl;
  }
  if ( world_rank == 0 )  std::cout << "AcceleratorHipInit: ================================================" << std::endl;
}
#endif


#ifdef GRID_SYCL

sycl::queue *theGridAccelerator;
sycl::queue *theCopyAccelerator;
void acceleratorInit(void)
{
  int nDevices = 1;
  //  sycl::gpu_selector selector;
  //  sycl::device selectedDevice { selector };
  theGridAccelerator = new sycl::queue (sycl::gpu_selector_v);
  theCopyAccelerator = new sycl::queue (sycl::gpu_selector_v);
  //  theCopyAccelerator = theGridAccelerator; // Should proceed concurrenlty anyway.

#ifdef GRID_SYCL_LEVEL_ZERO_IPC
  zeInit(0);
#endif
  
  char * localRankStr = NULL;
  int rank = 0;
  world_rank=0; 

  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv(ENV_LOCAL_RANK_OMPI)) != NULL)
  {
    rank = atoi(localRankStr);		
  }
  if ((localRankStr = getenv(ENV_LOCAL_RANK_MVAPICH)) != NULL)
  {
    rank = atoi(localRankStr);		
  }
  if ((localRankStr = getenv(ENV_LOCAL_RANK_PALS)) != NULL)
  {
    rank = atoi(localRankStr);		
  }
  if ((localRankStr = getenv(ENV_RANK_OMPI   )) != NULL) { world_rank = atoi(localRankStr);}
  if ((localRankStr = getenv(ENV_RANK_MVAPICH)) != NULL) { world_rank = atoi(localRankStr);}
  if ((localRankStr = getenv(ENV_RANK_PALS   )) != NULL) { world_rank = atoi(localRankStr);}

  char hostname[HOST_NAME_MAX+1];
  gethostname(hostname, HOST_NAME_MAX+1);
  if ( rank==0 ) std::cout << "AcceleratorSyclInit world_rank " << world_rank << " is host " << hostname << std::endl;

  auto devices = sycl::device::get_devices();
  for(int d = 0;d<devices.size();d++){

#define GPU_PROP(__prop__) \
    std::cout << "AcceleratorSyclInit:   " #__prop__ ": " << devices[d].get_info<sycl::info::device::__prop__>() << std::endl;

    if ( world_rank == 0) {

      GPU_PROP(vendor);
      GPU_PROP(version);
    //    GPU_PROP_STR(device_type);
    /*
    GPU_PROP(max_compute_units);
    GPU_PROP(native_vector_width_char);
    GPU_PROP(native_vector_width_short);
    GPU_PROP(native_vector_width_int);
    GPU_PROP(native_vector_width_long);
    GPU_PROP(native_vector_width_float);
    GPU_PROP(native_vector_width_double);
    GPU_PROP(native_vector_width_half);
    GPU_PROP(address_bits);
    GPU_PROP(half_fp_config);
    GPU_PROP(single_fp_config);
    */
    //    GPU_PROP(double_fp_config);
      GPU_PROP(global_mem_size);
    }

  }
  if ( world_rank == 0 ) {
    auto name = theGridAccelerator->get_device().get_info<sycl::info::device::name>();
    std::cout << "AcceleratorSyclInit: Selected device is " << name << std::endl;
    std::cout << "AcceleratorSyclInit: ================================================" << std::endl;
  }
}
#endif

#if (!defined(GRID_CUDA)) && (!defined(GRID_SYCL))&& (!defined(GRID_HIP))
void acceleratorInit(void){}
#endif

NAMESPACE_END(Grid);
