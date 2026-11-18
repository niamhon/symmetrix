#pragma once
#include <cmath>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// LAMMPS-style graph: owned atoms first, followed by distinct ghost slots.
// Image geometry stays on edges; equal IDs do not collapse image vectors.
class LSCGraph {
public:
    LSCGraph(std::span<const int> types,std::span<const int> senders,
             std::span<const int> receivers,std::span<const double> vectors,
             int num_owned=-1,std::span<const std::int64_t> ids={})
        : types_(types.begin(),types.end()),senders_(senders.begin(),senders.end()),
          receivers_(receivers.begin(),receivers.end()),vectors_(vectors.begin(),vectors.end()) {
        check(types.size()<=size_t(std::numeric_limits<int>::max()-1) && senders.size()<=size_t(std::numeric_limits<int>::max()),"excessive graph size");
        owned_=num_owned==-1?int(types.size()):num_owned;
        check(owned_>=0 && size_t(owned_)<=types.size(),"num_owned must be between zero and num_nodes");
        check(senders.size()==receivers.size() && vectors.size()==3*senders.size(),"expected edge_index[2,E] and edge_vectors[E,3]");
        check(ids.empty() || ids.size()==types.size(),"expected one atom ID per node slot");
        ids_.assign(ids.begin(),ids.end());if(ids.empty()){ids_.resize(types.size());for(size_t i=0;i<types.size();++i)ids_[i]=std::int64_t(i);}
        std::unordered_map<std::int64_t,int> owners,seen_types;
        for(size_t i=0;i<types.size();++i){check(types[i]>=0,"species index must be nonnegative");check(ids_[i]>=0,"atom ID must be nonnegative");
            auto [it,inserted]=seen_types.emplace(ids_[i],types[i]);check(inserted || it->second==types[i],"copies of an atom ID must have the same species");
            if(i<size_t(owned_))check(owners.emplace(ids_[i],int(i)).second,"owned atom IDs must be unique");}
        owner_.resize(types.size(),-1);
        for(size_t i=0;i<types.size();++i){auto found=owners.find(ids_[i]);if(found!=owners.end())owner_[i]=found->second;}
        for(size_t e=0;e<senders.size();++e){check(senders[e]>=0 && size_t(senders[e])<types.size(),"sender slot is out of range");
            check(receivers[e]>=0 && receivers[e]<owned_,"edge receivers must be owned atom slots");}
        for(double x:vectors)check((std::bit_cast<std::uint64_t>(x)&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000),"edge vectors must be finite");
    }
    int num_owned()const{return owned_;}
    int num_nodes()const{return int(types_.size());}
    int num_ghosts()const{return num_nodes()-owned_;}
    int num_edges()const{return int(senders_.size());}
    const std::vector<int>& types()const{return types_;}
    const std::vector<int>& senders()const{return senders_;}
    const std::vector<int>& receivers()const{return receivers_;}
    const std::vector<double>& edge_vectors()const{return vectors_;}
    const std::vector<std::int64_t>& atom_ids()const{return ids_;}
    const std::vector<int>& local_owners()const{return owner_;}
    std::span<const int> owned_types()const{return {types_.data(),size_t(owned_)};}
    bool has_remote_ghosts()const {for(int i=owned_;i<num_nodes();++i)if(owner_[i]<0)return true;return false;}
    bool uses_remote_ghosts()const {for(int i:senders_)if(owner_[i]<0)return true;return false;}
    std::vector<int> feature_senders()const {
        std::vector<int> out;out.reserve(senders_.size());
        for(int i:senders_)out.push_back(owner_[i]>=0?owner_[i]:i);
        return out;
    }
    std::vector<int> serial_senders()const {
        std::vector<int> out;out.reserve(senders_.size());
        for(int slot:senders_){check(owner_[slot]>=0,"remote ghost feature exchange is not implemented yet");out.push_back(owner_[slot]);}
        return out;
    }
private:
    static void check(bool valid,const char* message){if(!valid)throw std::invalid_argument(std::string("LSCGraph: ")+message);}
    int owned_=0;
    std::vector<int> types_,senders_,receivers_,owner_;
    std::vector<double> vectors_;
    std::vector<std::int64_t> ids_;
};
