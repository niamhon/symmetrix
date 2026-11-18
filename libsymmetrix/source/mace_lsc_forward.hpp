#pragma once
#include "mace_lsc_graph.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

// Shared CPU/CUDA bookkeeping; transport belongs to the caller (e.g. LAMMPS).
// Received feature rows overwrite ghosts. They are never summed or shifted.
class LSCForwardState {
public:
    void reset(const LSCGraph& graph) {
        for(auto& rows:ready_)rows.assign(graph.num_nodes(),false);
        available_={};complete_={};
    }
    void publish(int layer,const LSCGraph& graph) {
        const int t=index(layer);available_[t]=true;complete_[t]=false;
        auto& rows=ready_[t];rows.assign(graph.num_nodes(),false);
        for(int i=0;i<graph.num_nodes();++i)rows[i]=graph.local_owners()[i]>=0;
    }
    void check_available(int layer) const {
        if(!available_[index(layer)])fail("compute this layer before exchanging its features");
    }
    void check_pack(int layer,std::span<const int> slots,const LSCGraph& graph) const {
        check_available(layer);
        for(int i:slots){if(i<0 || i>=graph.num_nodes())fail("feature sender slot is out of range");
            if(!ready_[index(layer)][i])fail("cannot forward an unreceived ghost feature row");}
    }
    void check_receive(int layer,int first,std::span<const double> values,
                       const LSCGraph& graph,int width,double limit) const {
        check_available(layer);
        if(values.size()%width || first<graph.num_owned() || first>graph.num_nodes() ||
           values.size()/width>size_t(graph.num_nodes()-first))fail("received features must fit ghost rows");
        for(double x:values)if((std::bit_cast<std::uint64_t>(x)&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000) || x>limit || x< -limit)
            fail("received features must be finite and representable in the model precision");
    }
    void received(int layer,int first,int count) {
        for(int i=first;i<first+count;++i)ready_[index(layer)][i]=true;
        complete_[index(layer)]=false;
    }
    void complete(int layer,const LSCGraph& graph) {
        check_available(layer);
        for(int i:graph.senders())if(!ready_[index(layer)][i])
            fail(layer==1?"forward H1 exchange is incomplete":"forward H2 exchange is incomplete");
        complete_[index(layer)]=true;
    }
    bool complete(int layer) const {return complete_[index(layer)];}
private:
    static int index(int layer){if(layer!=1 && layer!=2)fail("expected layer 1 or 2");return layer-1;}
    [[noreturn]] static void fail(const char* s){throw std::invalid_argument(std::string("LSC forward: ")+s);}
    std::array<std::vector<bool>,2> ready_;
    std::array<bool,2> available_{},complete_{};
};
