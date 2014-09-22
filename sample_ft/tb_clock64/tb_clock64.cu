#include <stdio.h>

__global__
void kernel0(void) {
    printf("kernel0\n");
}

int main() {

    kernel0 <<<1,1>>> ();
    return 0;
}
