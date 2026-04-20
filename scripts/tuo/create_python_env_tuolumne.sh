#!/bin/bash

# module purge
# module load cpe/26.03
# rocm_version="7.2.0"
# module load PrgEnv-amd/8.7.0
# module load rocm/${rocm_version}
# module load craype-accel-amd-gfx942
# module load cray-python/3.10.10

# old versions to match frontier
module purge
module load cpe/24.11
rocm_version="6.2.4"
module load PrgEnv-cray
module load rocm/${rocm_version}
module load craype-accel-amd-gfx942
module load cray-python/3.10.10

# module load PrgEnv-cray
# rocm_version="7.2.0"
# module load rocm/${rocm_version}
# module load amd-mixed/${rocm_version}
# module load cray-mpich/9.1.0
# module load cpe/26.03
# module load craype-accel-amd-gfx942
# module load cray-python/3.10.10
# module load libtool

export ROCM_PATH="/opt/rocm-${rocm_version}/"

# change as needed
export WRKSPC=/usr/WS1/$USER/distributed-gnn/plexus/
mkdir -p $WRKSPC
cd $WRKSPC

# change as needed
ENV_NAME="my-venv"
ENV_LOC="$WRKSPC/$ENV_NAME"


# Setup Virtual Environment
echo "Setting up Virtual Environment"
uv venv ${ENV_LOC}
. ${ENV_LOC}/bin/activate


# Python Packages
uv pip install --upgrade pip

echo "Installing PyTorch"
if [ "${rocm_version}" == 5.6.0  ]; then
	uv pip install --force-reinstall /lustre/orion/world-shared/stf007/msandov1/wheels/TorchROCm5.6/torch-2.1.2-cp310-cp310-linux_x86_64.whl
elif [ "${rocm_version}" == 6.0.0  ]; then
	uv pip install torch  --index-url https://download.pytorch.org/whl/rocm6.0
elif [ "${rocm_version}" == 5.7.0  ]; then
	uv pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/rocm5.7
elif [ "${rocm_version}" == 6.2.4  ]; then
	uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2.4
	uv pip install --upgrade numpy
elif [ "${rocm_version}" == 7.2.0  ]; then
	uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
	uv pip install --upgrade numpy
fi

uv pip install torch_geometric
uv pip install numpy
#uv pip install axonn
uv pip install /g/g16/dnicho/distributed-gnn/plexus/axonn
uv pip install ogb


# RCCL plugin
echo "Installing RCCL Plugin"
if [ -d "aws-ofi-nccl" ]; then
	rm -rf aws-ofi-nccl
fi
git clone --recursive --depth=1 https://github.com/aws/aws-ofi-nccl
cd aws-ofi-nccl
libfabric_path=/opt/cray/libfabric/2.1
./autogen.sh
export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/lib:$LD_LIBRARY_PATH
CC=cc CFLAGS=-I/opt/rocm-$rocm_version/include ./configure \
    --with-libfabric=$libfabric_path --enable-trace \
    --prefix=$PWD --with-rocm=/opt/rocm-$rocm_version --with-mpi=$MPICH_DIR
make
make install
ln -s $PWD/lib/librccl-net.so $PWD/lib/libnccl-net.so
ln -s $PWD/lib/librccl-net-ofi.so $PWD/lib/libnccl-net-ofi.so
ln -s $PWD/lib/librccl-net.la $PWD/lib/libnccl-net.la
ln -s $PWD/lib/librccl-net-ofi.la $PWD/lib/libnccl-net-ofi.la
cd ..
tar -czf aws-ofi-nccl.tar.gz aws-ofi-nccl/