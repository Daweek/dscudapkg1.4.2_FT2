#include <stdio.h>
#include <stdlib.h>


__global__ void
gpuStress() {
}

int main() {
    const int t_max = 10000;

    for ( t=0; t<t_max; i++ ){
	gpuStress <<< , >>> (  );
    }
    return EXIT_SUCCESS;
}
