#ifndef VERSION_H
#define VERSION_H

//
// ver2.0   add function for counting inside cutoff distance calculation.
//

#define MAJOR_VER 2
#define MINOR_VER 0
#define BUILD_CNT 0

void printVersion(void)
{
  printf("Build Version: %d.%d (%d)\n", MAJOR_VER, MINOR_VER, BUILD_CNT);
}
#endif /*VERSION_H*/
