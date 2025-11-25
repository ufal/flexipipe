#ifndef IO_TEITOK_WRITER_H
#define IO_TEITOK_WRITER_H

#include <string>
#include <vector>
#include <ostream>
#include "types.h"

class TEITOKWriter {
public:
    // Write sentences to TEITOK XML format
    static void write(const std::vector<Sentence>& sentences, std::ostream& out);
    
    // Write to file
    static bool write_file(const std::vector<Sentence>& sentences, const std::string& file_path);
};

#endif // IO_TEITOK_WRITER_H

