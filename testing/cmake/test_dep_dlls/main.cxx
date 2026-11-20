#ifdef STATIC_CONSUMER
int static_dependency();
int main() { return static_dependency(); }
#else
int main() { return 0; }
#endif
