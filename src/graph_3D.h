/*
 file graph_3D.h
 */


#ifndef GRAPH_3D_H
#define GRAPH_3D_H

#include "includeDef.h"
//#include <list>
//#include <array>
//#include <vector>
#include <string>
#include <map>

using namespace arma;

static string surf_figs_ind_file_name("surf_figs_ind.dat");

extern "C++"
{
	typedef map<string,string> options_list;
	
    class graph_3D
    {
    public:
        graph_3D(int =0);
        ~graph_3D();
        
		void add_data(vec x, vec y, mat Z);
		void plot_surface(vec x, vec y, mat Z, map<string,string> extra_options={}, int fig_ind_p=0);
		void plot_surface(map<string,string> extra_options={}, int fig_ind_p=0);
		void set_axes_lims(double *xlims_par, double *ylims_par, double *zlims_par);
        void set_axes_labels(const char*, const char*, const char*);
		void add_title(const char *ttl);
        
        double labels_fontsize;
        double legend_fontsize;
		double title_fontsize;
        int legend_loc;
		
		static void open_pipe();
		static void close_pipe();
        static void show_figures();
		static void show_commands(bool show_comm){show_command=show_comm;}
		static void reset_figs_ind_file(){if (figs_ind_file) figs_ind_file.close(); figs_ind_file.open(surf_figs_ind_file_name); file_ind=0;}
		
		static char plot_command_format[500];
        static FILE *plot_pipe;
		static FILE *file_pipe;
		static bool display_figures;
		static bool print_to_file;
        static int file_ind;
        static int fig_ind;
        static char program_name[100];
        static char config_command[200];
        static char fig_command_format[100];
        static char load_x_command_format[100];
		static char load_y_command_format[100];
		static char load_z_command_format[100];
		static char mesh_command[100];
		static char title_command_format[100];
        static char xlabel_command_format[100];
        static char ylabel_command_format[100];
		static char zlabel_command_format[100];
        static char xlims_command_format[100];
        static char ylims_command_format[100];
		static char zlims_command_format[100];
        static char show_figures_command[100];
		static char x_file_name_format[100];
		static char y_file_name_format[100];
		static char z_file_name_format[100];
		static bool show_command;
		static char file_name[100];
		static char file_name_format[100];
		static char figs_dir[100];
		static ofstream figs_ind_file;
        
    private:
        char title[400];
        char xlabel[100];
        char ylabel[100];
		char zlabel[100];

        double xlims[2];
        double ylims[2];
		double zlims[2];
		
		vec x,y;
		mat Z;
    };
    
}

#endif
