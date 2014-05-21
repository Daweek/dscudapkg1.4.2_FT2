#ifndef INITIAL_CONDITION_H
#define INITIAL_CONDITION_H


void   printUsage(char **);
void   getLineFromFile(char *, const char *, const char *);
//
void getPathToSaveRoot(char *path);
void getPathToNthRemd(char *path, int rep_i);
void getPathToLocal(char *path);

void initSimConfig(int, char **);
void echoSimConfig(void);
//
void loadValFromFile(double  &, const char *, const char *);
void loadValFromFile(int     &, const char *, const char *);
void loadValFromFile(long    &, const char *, const char *);
void loadValFromFile(float   &, const char *, const char *);
void loadValFromFile(char    *, const char *, const char *);

#endif // INITIAL_CONDITION_H
