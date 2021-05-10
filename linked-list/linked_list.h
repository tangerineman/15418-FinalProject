

class Node {
    public:
        Node *prev;
        Node *next;

        float x;
        float y;

        float r;
        float g;
        float b;
        float a; 
};

class List {
    public:
        Node *head;
        Node *tail;
        int size;
};

__device__ List *linked_list_init();

__device__ void insert_node(List* list, float x, float y, float r, float g, float b, float a);
