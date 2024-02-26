# Specify the target executable and the source files needed to build it
symnmf: matrix.o symnmf.o matrix.h symnmf.h
	gcc -o symnmf matrix.o symnmf.o -lm

# Specify the object files that are generated from the corresponding source files
matrix.o: matrix.c
	gcc -ansi -Wall -Wextra -Werror -pedantic-errors -c matrix.c

symnmf.o: symnmf.c
	gcc -ansi -Wall -Wextra -Werror -pedantic-errors -c symnmf.c 

