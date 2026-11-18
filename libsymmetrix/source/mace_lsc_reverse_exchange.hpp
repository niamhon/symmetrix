#pragma once
#include "mace_lsc_charge_exchange.hpp"
// A reverse SUM consumes ghost rows once. Blocking MPI callers complete all
// receives before advancing. Intermediate ghost relays are valid destinations.
class LSCReverseExchange {
public:
    int phase=0;
    void reset(){phase=0;sent_.clear();}
    void begin(const LSCGraph& g){phase=1;sent_.assign(g.num_nodes(),false);}
    void require_phase(int p)const{check(phase==p,"reverse phase is not ready");}
    void next(int from,int to,const LSCGraph& g){require_phase(from);phase=to;sent_.assign(g.num_nodes(),false);}
    void check_pack(int p,int first,int count,int width,std::span<double> values,const LSCGraph& g)const{
        require_phase(p);check(count>=0 && first>=g.num_owned() && first<=g.num_nodes() && count<=g.num_nodes()-first,"reverse send must fit ghost rows");
        check(values.size()==size_t(count)*width,"wrong reverse packet width");
        for(int i=first;i<first+count;++i)check(!sent_[i],"ghost reverse row was already returned");
    }
    void sent(int first,int count){for(int i=first;i<first+count;++i)sent_[i]=true;}
    void check_add(int p,std::span<const int> slots,int width,std::span<const double> values,const LSCGraph& g)const{
        require_phase(p);check(values.size()==slots.size()*width,"wrong reverse receive width");LSCChargeExchange::finite(values);
        for(size_t k=0;k<slots.size();++k){int i=slots[k];check(i>=0 && i<g.num_nodes(),"reverse receive slot out of range");
            if(sent_[i])for(int d=0;d<width;++d)check(values[k*width+d]==0,"received derivative after its ghost row was returned");}
    }
    void finish(int p,const LSCGraph& g){require_phase(p);for(int i:g.senders())if(g.local_owners()[i]<0)check(sent_[i],"reverse exchange is incomplete");next(p,p+1,g);}
    static void check(bool ok,const char* text){if(!ok)throw std::invalid_argument(std::string("LSC reverse exchange: ")+text);}
private:
    std::vector<bool> sent_;
};
