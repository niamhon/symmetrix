#pragma once
#include "mace_lsc_forward.hpp"
#include <algorithm>
#include <limits>

// Reverse charge messages add two learned monopoles (one per source layer).
// Forward potential messages copy full dE_electrostatic/d[q,dy,dz,dx].
class LSCChargeExchange {
public:
    void reset(){stage_=0;packed_.clear();potentials_=LSCForwardState{};}
    void begin(const LSCGraph& graph){reset();stage_=1;packed_.assign(graph.num_nodes(),false);}
    bool active()const{return stage_!=0;}
    bool partial()const{return stage_==1;}
    bool charges_ready()const{return stage_>=2;}
    std::string stage()const{return stage_==0?"empty":stage_==1?"partial_charges":stage_==2?"assembled_charges":stage_==3?"potentials":"potentials_complete";}
    static void finite(std::span<const double> values){for(double x:values)
        if((std::bit_cast<std::uint64_t>(x)&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000))fail("expected finite charge/potential values");}
    void check_pack(int first,int count,const LSCGraph& graph)const{
        require(partial(),"compute partial charges before reverse exchange");
        require(first>=graph.num_owned() && count>=0 && first<=graph.num_nodes() && count<=graph.num_nodes()-first,"reverse sends must fit ghost rows");
        for(int i=first;i<first+count;++i)require(!packed_[i],"ghost charge row was already returned");
    }
    void sent(int first,int count){for(int i=first;i<first+count;++i)packed_[i]=true;}
    void check_add(std::span<const int> slots,std::span<const double> values,const LSCGraph& graph)const{
        require(partial(),"reverse charge exchange is not active");
        require(values.size()==2*slots.size(),"expected two partial charges per receive row");finite(values);
        for(size_t k=0;k<slots.size();++k){int i=slots[k];require(i>=0 && i<graph.num_nodes(),"reverse receive slot is out of range");
            require(!packed_[i] || (values[2*k]==0 && values[2*k+1]==0),"received charge after its ghost row was returned");}
    }
    void finish(const LSCGraph& graph){
        require(partial(),"complete charge exchange exactly once");
        for(int i:graph.senders())if(graph.local_owners()[i]<0)require(packed_[i],"partial-charge reverse exchange is incomplete");
        stage_=2;
    }
    void set_potentials(const LSCGraph& graph,std::span<const double> values){
        require(charges_ready(),"assemble owned charges before setting electrostatic potentials");
        require(values.size()==size_t(graph.num_owned())*4,"expected owned potentials[N_owned,4]");finite(values);
        potentials_.reset(graph);potentials_.publish(1,graph);stage_=3;
    }
    void check_potential_pack(std::span<const int> slots,const LSCGraph& graph)const{
        require(stage_>=3,"set owned electrostatic potentials first");potentials_.check_pack(1,slots,graph);
    }
    void check_potential_receive(int first,std::span<const double> values,const LSCGraph& graph)const{
        require(stage_==3,"receive potentials before completing their exchange");
        potentials_.check_receive(1,first,values,graph,4,std::numeric_limits<double>::max());
    }
    void received_potentials(int first,int count){potentials_.received(1,first,count);}
    void complete_potentials(const LSCGraph& graph){require(stage_==3,"set and exchange potentials before completion");try{potentials_.complete(1,graph);}catch(const std::invalid_argument&){fail("electrostatic potential exchange is incomplete");}stage_=4;}
    bool potentials_ready()const{return stage_==4;}
    bool potentials_available()const{return stage_>=3;}
    static void require(bool ok,const char* s){if(!ok)fail(s);}
private:
    [[noreturn]] static void fail(const char* s){throw std::invalid_argument(std::string("LSC charge exchange: ")+s);}
    int stage_=0;
    std::vector<bool> packed_;
    LSCForwardState potentials_;
};
struct LSCSlotAdjacency {
    std::vector<int> incoming_starts,incoming_edges,outgoing_starts,outgoing_edges;
    explicit LSCSlotAdjacency(const LSCGraph& graph){
        const auto senders=graph.feature_senders();const auto& receivers=graph.receivers();
        incoming_starts.assign(size_t(graph.num_nodes())+1,0);outgoing_starts=incoming_starts;
        for(size_t e=0;e<senders.size();++e){++incoming_starts[receivers[e]+1];++outgoing_starts[senders[e]+1];}
        for(int i=0;i<graph.num_nodes();++i){incoming_starts[i+1]+=incoming_starts[i];outgoing_starts[i+1]+=outgoing_starts[i];}
        incoming_edges.resize(senders.size());outgoing_edges.resize(senders.size());
        auto ic=incoming_starts,oc=outgoing_starts;
        for(int e=0;e<graph.num_edges();++e){incoming_edges[ic[receivers[e]]++]=e;outgoing_edges[oc[senders[e]]++]=e;}
    }
};
