// Staged integration check using LAMMPS communication. The global direct
// electrostatic reference here is diagnostic only, not a production solver.
#include "lammps.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "force.h"
#include "input.h"
#include "neighbor.h"
#include "update.h"
#include "pair_symmetrix_mace_lsc.h"
#include "json.h"
#include "mace_lsc.hpp"
#include <algorithm>
#include <cstdint>
#include <mpi.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace LAMMPS_NS;
using json=LAMMPS_NS::json;

template<bool CUDA> json forward_result(LAMMPS& lmp,const std::string& source_model,bool reverse,bool mesh,bool serial)
{
    auto pair=dynamic_cast<PairSymmetrixMACELSC<CUDA>*>(lmp.force->pair);
    if(!pair)throw std::runtime_error("Unexpected pair style");
    if(source_model.empty())pair->compute_forward();else pair->compute_distributed_sources();
    json result;
    result["rank"]=lmp.comm->me;result["ranks"]=lmp.comm->nprocs;
    result["num_owned"]=lmp.atom->nlocal;result["num_ghosts"]=lmp.atom->nghost;
    std::vector<std::int64_t> ids(lmp.atom->nlocal+lmp.atom->nghost);
    for(size_t i=0;i<ids.size();++i)ids[i]=lmp.atom->tag[i];
    result["atom_ids"]=ids;result["H1"]=pair->forward_features(1);result["H2"]=pair->forward_features(2);
    result["local_node_energies"]=pair->forward_node_energies();
    result["local_energy"]=pair->forward_local_energy();
    if(serial){
        for(int i=0;i<lmp.atom->nlocal+lmp.atom->nghost;++i)std::fill_n(lmp.atom->f[i],3,0.);
        pair->compute(1,1);
        result["density_coefficients"]=pair->owned_density();result["dipole"]=pair->source_dipole();result["forces"]=pair->owned_forces();
        result["energy"]=pair->eng_vdwl;result["virial"]=std::vector<double>(pair->virial,pair->virial+6);
        double volume=lmp.domain->xprd*lmp.domain->yprd*lmp.domain->zprd;
        const int order[6]={0,1,2,5,4,3};std::array<double,6> stress{};for(int d=0;d<6;++d)stress[d]=-pair->virial[order[d]]/volume;
        result["stress"]=stress;
    }else if(mesh){
        // Only one set of local bounds is initialized by Domain. Poison the
        // inactive set so plan-cache reuse cannot accidentally depend on it.
        auto lo=lmp.domain->triclinic?lmp.domain->sublo:lmp.domain->sublo_lamda;
        auto hi=lmp.domain->triclinic?lmp.domain->subhi:lmp.domain->subhi_lamda;
        std::fill_n(lo,3,std::numeric_limits<double>::quiet_NaN());
        std::fill_n(hi,3,std::numeric_limits<double>::quiet_NaN());
        result["unused_bounds_poisoned"]=true;
        pair->evaluate_distributed_mesh();
        result["density_coefficients"]=pair->owned_density();result["dipole"]=pair->source_dipole();
        result["potentials"]=pair->density_potentials();result["forces"]=pair->owned_forces();
        result["energy"]=pair->distributed_energy;result["virial"]=pair->distributed_virial;result["stress"]=pair->distributed_stress;
        result["mesh_energy_rank"]=pair->mesh_result.energy;result["mesh_position_gradient"]=pair->mesh_result.position_gradient;
        result["mesh_strain_rank"]=pair->mesh_result.strain_derivative;result["mesh_shape"]=pair->mesh_shape;
        result["mesh_local_modes"]=pair->mesh_local_modes;result["mesh_local_brick"]=pair->mesh_local_brick;
        result["mesh_device_fft_calls"]=pair->mesh_device_fft_calls;result["mesh_plan_generation"]=pair->mesh_plan_generation;
        // A second solve on the same graph must reuse the distributed plans.
        pair->evaluate_distributed_mesh();result["mesh_reused_plan_generation"]=pair->mesh_plan_generation;
        result["mesh_repeat_energy"]=pair->distributed_energy;
    }else if(!source_model.empty()){
        // Diagnostic only: assemble a small global density and solve the
        // existing serial direct operator. Production LSC adds no allgather
        // or replicated Coulomb solver in this communication stage.
        const int n=lmp.atom->nlocal,size=lmp.comm->nprocs;
        std::vector<int> counts(size),offsets(size),counts4(size),offsets4(size),counts3(size),offsets3(size);
        MPI_Allgather(&n,1,MPI_INT,counts.data(),1,MPI_INT,MPI_COMM_WORLD);
        int total=0;for(int i=0;i<size;++i){offsets[i]=total;total+=counts[i];counts4[i]=4*counts[i];counts3[i]=3*counts[i];offsets4[i]=4*offsets[i];offsets3[i]=3*offsets[i];}
        const auto rho=pair->owned_density();std::vector<double> xyz(size_t(n)*3),all_rho(size_t(total)*4),all_xyz(size_t(total)*3);
        for(int i=0;i<n;++i){double unwrapped[3];lmp.domain->unmap(lmp.atom->x[i],lmp.atom->image[i],unwrapped);
            std::copy_n(unwrapped,3,xyz.data()+size_t(i)*3);}
        double dummy=0;
        MPI_Allgatherv(rho.empty()?&dummy:rho.data(),4*n,MPI_DOUBLE,all_rho.empty()?&dummy:all_rho.data(),counts4.data(),offsets4.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgatherv(xyz.empty()?&dummy:xyz.data(),3*n,MPI_DOUBLE,all_xyz.empty()?&dummy:all_xyz.data(),counts3.data(),offsets3.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        const std::array<double,9> box={lmp.domain->xprd,0,0,lmp.domain->xy,lmp.domain->yprd,0,lmp.domain->xz,lmp.domain->yz,lmp.domain->zprd};
        const std::array<bool,3> pbc={bool(lmp.domain->xperiodic),bool(lmp.domain->yperiodic),bool(lmp.domain->zperiodic)};
        MACELSC reference(source_model);
        const auto direct=reference.evaluate_direct_electrostatics(all_xyz,all_rho,box,pbc,"auto");
        std::vector<double> owned_phi(direct.density_gradient.begin()+offsets4[lmp.comm->me],direct.density_gradient.begin()+offsets4[lmp.comm->me]+4*n);
        pair->exchange_electrostatic_potentials(owned_phi);
        result["density_coefficients"]=rho;result["dipole"]=pair->source_dipole();
        result["potentials"]=pair->density_potentials();result["reference_owned_potentials"]=owned_phi;
        result["reference_electrostatic_energy"]=direct.energy;
        if(reverse){
            std::vector<double> pg(direct.position_gradient.begin()+offsets3[lmp.comm->me],direct.position_gradient.begin()+offsets3[lmp.comm->me]+3*n);
            auto strain=direct.strain_derivative;for(double& x:strain)x/=size;
            // Test-only partition of the replicated operator's energy/strain.
            // A production distributed solver must supply its own contributions.
            double el=0;for(size_t k=0;k<rho.size();++k)el+=.5*rho[k]*owned_phi[k];
            pair->reverse_distributed(pg,strain,box,el);
            result["forces"]=pair->owned_forces();result["energy"]=pair->distributed_energy;
            result["virial"]=pair->distributed_virial;result["stress"]=pair->distributed_stress;
        }
    }
    return result;
}
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);
    int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    try {
        if(argc!=4 && argc!=6)throw std::runtime_error("Expected setup.in output-prefix backend [charges|reverse|mesh|serial model.json]");
        if(argc==6 && std::string(argv[4])!="charges" && std::string(argv[4])!="reverse" && std::string(argv[4])!="mesh" && std::string(argv[4])!="serial")throw std::runtime_error("Expected charges, reverse or mesh stage selector");
        const std::string source_model=argc==6?argv[5]:"";
        const bool reverse=argc==6 && std::string(argv[4])=="reverse";
        const bool mesh=argc==6 && std::string(argv[4])=="mesh";
        const bool serial=argc==6 && std::string(argv[4])=="serial";
        char name[]="lsc-forward-check",screen[]="-screen",none[]="none",log[]="-log";
        char* args[]={name,screen,none,log,none};
        {
            LAMMPS lmp(5,args,MPI_COMM_WORLD);
            lmp.input->file(argv[1]);
            lmp.update->whichflag=1;
            lmp.update->firststep=lmp.update->laststep=lmp.update->ntimestep;
            lmp.update->beginstep=lmp.update->endstep=lmp.update->ntimestep;
            lmp.init();
            // Match Verlet setup through neighbor construction, omitting force
            // evaluation/integration/output. Test inputs contain no fixes.
            lmp.update->setupflag=1;
            lmp.atom->setup();
            if(lmp.domain->triclinic)lmp.domain->x2lamda(lmp.atom->nlocal);
            lmp.domain->pbc();lmp.domain->reset_box();lmp.comm->setup();
            if(lmp.neighbor->style)lmp.neighbor->setup_bins();
            lmp.comm->exchange();lmp.comm->borders();
            if(lmp.domain->triclinic)lmp.domain->lamda2x(lmp.atom->nlocal+lmp.atom->nghost);
            lmp.neighbor->build(1);
            json result;
            if(std::string(argv[3])=="cpu")result=forward_result<false>(lmp,source_model,reverse,mesh,serial);
#ifdef SYMMETRIX_CUDA
            else result=forward_result<true>(lmp,source_model,reverse,mesh,serial);
#else
            else throw std::runtime_error("Driver built without CUDA pair style");
#endif
            std::ofstream file(std::string(argv[2])+".rank"+std::to_string(rank)+".json");
            file<<result.dump()<<'\n';
            if(!file)throw std::runtime_error("Cannot write MPI check output");
            lmp.update->setupflag=0;
        }
        MPI_Finalize();return 0;
    } catch(const std::exception& e){
        std::cerr<<"LSC_FORWARD_CHECK rank "<<rank<<": "<<e.what()<<'\n';
        MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
