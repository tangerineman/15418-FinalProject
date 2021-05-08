
#include <stdio.h>
#include <cuda.h>
#include "linked_list.h"



// returns empty linked list with just a head (head contains no information)
__device__ List *linked_list_init(){
    List *newList = new(List);

    Node *newNode = new (Node);
    newNode->next = NULL;
    newNode->prev = NULL;
    newNode->x = NULL;
    newNode->y = NULL;
    newNode->r = NULL;
    newNode->g = NULL;
    newNode->b = NULL;
    newNode->a = NULL;

    newList->head = newNode;
    newList->tail = newNode;
    newList->size = 1;

    return newList;
}

// inserts a node at the tail of the linked list
__device__ void insert_node(List* list, float x, float y, float r, float g, float b, float a){

    Node *newNode = new (Node);
 
    newNode->prev = list->tail;
    newNode->next = NULL;
    newNode->x = x;
    newNode->y = y;
    
    newNode->r = r;
    newNode->g = g;
    newNode->b = b;
    newNode->a = a; 

    list->tail->next = newNode;
    list->tail = newNode;
    list->size++;
    // printf("size = %d", list->size);
    return;

}

