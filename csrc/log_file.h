/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>


inline std::string getCurrentDateTime( std::string s ){
    time_t now = time(0);
    struct tm  tstruct;
    char  buf[80];
    tstruct = *localtime(&now);
    if(s=="now")
        strftime(buf, sizeof(buf), "%Y-%m-%d %X ", &tstruct);
    else if(s=="date")
        strftime(buf, sizeof(buf), "%Y-%m-%d ", &tstruct);
    return  std::string(buf);
};


struct output_stream{
    // output_stream(std::string file_name ) :file(file_name) {}
    output_stream(std::string file_name = "out.txt")  {}
    // output_stream()
    template < typename T > output_stream& operator<< ( const T& value ) {
        // std::cout << getCurrentDateTime("now")  <<value ;
        std::cout  <<value ;
        file << getCurrentDateTime("now")   << value ;
        // file.flush();
        return *this ;
    }

    // ios manipulators
    output_stream& operator<< ( std::ios_base& (*manip) ( std::ios_base& ) ) {
        std::cout << manip ;
        file  << manip ;
        // file.flush();
        return *this ;
    }

    template < typename... T > // ostream manipulators with zero or more arguments
    output_stream& operator<< ( std::ostream& (*manip) (  std::ostream&, const T&... )  ) {

        std::cout << manip ;
        file   << manip ;
        // file.flush();
        return *this ;
    }

    void init(){
        file = std::ofstream(file_name, std::ios_base::app);
        file << "Experiment running time ";
        file << getCurrentDateTime("now") << std::endl;
    }
    void close(){
        file.close();
    }
    std::string file_name;
    std::ofstream file;
};

