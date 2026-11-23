
#ifndef TOK_H
#define TOK_H

#include <stddef.h>
#include "hashmap.h"


//Much like a python split function, but in this case, you should also split on punctuation.
//Unlike split() however, whitespace (" ", "\t", "\n") are tracked as their own tokens
//punctuation also is considered its own token.
// Ex: eax, -> ["eax", ","]
char **split_tokens(char *content);


//key: token str, value: integer ID --> if not seen, assign id, counter++;
//Why have an id? It will be the row index in the embedding matrix for the given token.
void encode_tokens(char **toks, struct hashmap_s *token_encode_map);
// either that or do it this way: struct hashmap_s encode_tokens(char **toks);


//returns an arr of the retrieved token ids in order given input **toks
int *get_token_ids(struct hashmap_s *token_encode_map, char **toks, size_t num_tokens);


//needed before we can do split_tokens on ex: train.txt
char *read_file_content(char *path);
//important note. We also need a train_test_split on ryans data
//you can either make some C logic or use scikitlearn in a quick python script

//optional / python prob better
void train_test_split(char *path);



#endif //TOK_H