/* generated by common/SynapseHeader.h.jinja2 */ /**
 *  stdp_stp__with_iaf_psc_exp_wta.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Generated from NESTML at time: 2022-09-07 12:29:38.676056
**/

#ifndef STDP_STP__WITH_IAF_PSC_EXP_WTA_H
#define STDP_STP__WITH_IAF_PSC_EXP_WTA_H

// C++ includes:
#include <cmath>

// Includes from nestkernel:
#include "common_synapse_properties.h"
#include "connection.h"
#include "connector_model.h"
#include "event.h"

// Includes for random number generator
#include <random>


// Includes from sli:
#include "dictdatum.h"
#include "dictutils.h"

/** @BeginDocumentation

**/

#define POST_NEURON_TYPE iaf_psc_exp_wta__with_stdp_stp

//#define DEBUG

namespace nest
{

namespace stdp_stp__with_iaf_psc_exp_wta_names
{
    const Name _w( "w" );
    const Name _u( "u" );
    const Name _x( "x" );
    const Name _decay_time_kernel__X__pre_spikes( "decay_time_kernel__X__pre_spikes" );
    const Name _rise_time_kernel__X__pre_spikes( "rise_time_kernel__X__pre_spikes" );
    const Name _d( "d" );
    const Name _U( "U" );
    const Name _tau_d( "tau_d" );
    const Name _tau_f( "tau_f" );
    const Name _tau_rise( "tau_rise" );
    const Name _tau_decay( "tau_decay" );
    const Name _c( "c" );
}

class stdp_stp__with_iaf_psc_exp_wtaCommonSynapseProperties : public CommonSynapseProperties {
public:

    stdp_stp__with_iaf_psc_exp_wtaCommonSynapseProperties()
    : CommonSynapseProperties()
    {
    }

    /**
     * Get all properties and put them into a dictionary.
     */
    void get_status( DictionaryDatum& d ) const
    {
        CommonSynapseProperties::get_status( d );
    }


    /**
     * Set properties from the values given in dictionary.
     */
    void set_status( const DictionaryDatum& d, ConnectorModel& cm )
    {
      CommonSynapseProperties::set_status( d, cm );
    }

    // N.B.: we define all parameters as public for easy reference conversion later on.
    // This may or may not benefit performance (TODO: compare with inline getters/setters)
};


template < typename targetidentifierT >
class stdp_stp__with_iaf_psc_exp_wta : public Connection< targetidentifierT >
{
private:
  double t_lastspike_;

  /**
   * Dynamic state of the synapse.
   *
   * These are the state variables that are advanced in time by calls to
   * send(). In many models, some or all of them can be set by the user
   * through ``SetStatus()``.
   *
   * @note State_ need neither copy constructor nor @c operator=(), since
   *       all its members are copied properly by the default copy constructor
   *       and assignment operator. Important:
   *       - If State_ contained @c Time members, you need to define the
   *         assignment operator to recalibrate all members of type @c Time . You
   *         may also want to define the assignment operator.
   *       - If State_ contained members that cannot copy themselves, such
   *         as C-style arrays, you need to define the copy constructor and
   *         assignment operator to copy those members.
  **/
  struct State_{    /* generated by directives/MemberDeclaration.jinja2 */ 
    //!  1.?
    double w;/* generated by directives/MemberDeclaration.jinja2 */ 
    //!  facilitation
    double u;/* generated by directives/MemberDeclaration.jinja2 */ 
    //!  depression
    double x;/* generated by directives/MemberDeclaration.jinja2 */ 
    double decay_time_kernel__X__pre_spikes;/* generated by directives/MemberDeclaration.jinja2 */ 
    double rise_time_kernel__X__pre_spikes;

    State_() {};
  };

  /**
   * Free parameters of the synapse.
   *
   *
   *
   * These are the parameters that can be set by the user through @c SetStatus.
   * Parameters do not change during calls to ``send()`` and are not reset by
   * @c ResetNetwork.
   *
   * @note Parameters_ need neither copy constructor nor @c operator=(), since
   *       all its members are copied properly by the default copy constructor
   *       and assignment operator. Important:
   *       - If Parameters_ contained @c Time members, you need to define the
   *         assignment operator to recalibrate all members of type @c Time . You
   *         may also want to define the assignment operator.
   *       - If Parameters_ contained members that cannot copy themselves, such
   *         as C-style arrays, you need to define the copy constructor and
   *         assignment operator to copy those members.
  */
  struct Parameters_{    /* generated by directives/MemberDeclaration.jinja2 */ 
    //!  !!! cannot have a variable called "delay"
    //!  STP
    double d;/* generated by directives/MemberDeclaration.jinja2 */ 
    //!  STP
    double U;/* generated by directives/MemberDeclaration.jinja2 */ 
    //!  depression timescale
    double tau_d;/* generated by directives/MemberDeclaration.jinja2 */ 
    //!  facilitation timescale
    //!  STDP
    double tau_f;/* generated by directives/MemberDeclaration.jinja2 */ 
    //!  STDP
    double tau_rise;/* generated by directives/MemberDeclaration.jinja2 */ 
    double tau_decay;/* generated by directives/MemberDeclaration.jinja2 */ 
    double c;



    /** Initialize parameters to their default values. */
    Parameters_() {};
  };

  /**
   * Internal variables of the synapse.
   *
   *
   *
   * These variables must be initialized by recompute_internal_variables().
  **/
  struct Variables_
  {    /* generated by directives/MemberDeclaration.jinja2 */ 
    double __h;    /* generated by directives/MemberDeclaration.jinja2 */ 
    double __P__u__u;    /* generated by directives/MemberDeclaration.jinja2 */ 
    double __P__x__x;    /* generated by directives/MemberDeclaration.jinja2 */ 
    double __P__decay_time_kernel__X__pre_spikes__decay_time_kernel__X__pre_spikes;    /* generated by directives/MemberDeclaration.jinja2 */ 
    double __P__rise_time_kernel__X__pre_spikes__rise_time_kernel__X__pre_spikes;
  };

  Parameters_ P_;  //!< Free parameters.
  State_      S_;  //!< Dynamic state.
  Variables_  V_;  //!< Internal Variables

  // -------------------------------------------------------------------------
  //   Getters/setters for state block
  // -------------------------------------------------------------------------

  inline double get_w() const
  {
    return S_.w;
  }

  inline void set_w(const double __v)
  {
    S_.w = __v;
  }

  inline double get_u() const
  {
    return S_.u;
  }

  inline void set_u(const double __v)
  {
    S_.u = __v;
  }

  inline double get_x() const
  {
    return S_.x;
  }

  inline void set_x(const double __v)
  {
    S_.x = __v;
  }

  inline double get_decay_time_kernel__X__pre_spikes() const
  {
    return S_.decay_time_kernel__X__pre_spikes;
  }

  inline void set_decay_time_kernel__X__pre_spikes(const double __v)
  {
    S_.decay_time_kernel__X__pre_spikes = __v;
  }

  inline double get_rise_time_kernel__X__pre_spikes() const
  {
    return S_.rise_time_kernel__X__pre_spikes;
  }

  inline void set_rise_time_kernel__X__pre_spikes(const double __v)
  {
    S_.rise_time_kernel__X__pre_spikes = __v;
  }


  // -------------------------------------------------------------------------
  //   Getters/setters for parameters
  // -------------------------------------------------------------------------

  inline double get_d() const
  {
    return P_.d;
  }

  inline void set_d(const double __v)
  {
    P_.d = __v;
  }

  inline double get_U() const
  {
    return P_.U;
  }

  inline void set_U(const double __v)
  {
    P_.U = __v;
  }

  inline double get_tau_d() const
  {
    return P_.tau_d;
  }

  inline void set_tau_d(const double __v)
  {
    P_.tau_d = __v;
  }

  inline double get_tau_f() const
  {
    return P_.tau_f;
  }

  inline void set_tau_f(const double __v)
  {
    P_.tau_f = __v;
  }

  inline double get_tau_rise() const
  {
    return P_.tau_rise;
  }

  inline void set_tau_rise(const double __v)
  {
    P_.tau_rise = __v;
  }

  inline double get_tau_decay() const
  {
    return P_.tau_decay;
  }

  inline void set_tau_decay(const double __v)
  {
    P_.tau_decay = __v;
  }

  inline double get_c() const
  {
    return P_.c;
  }

  inline void set_c(const double __v)
  {
    P_.c = __v;
  }



  // -------------------------------------------------------------------------
  //   Getters/setters for inline expressions
  // -------------------------------------------------------------------------
  
  inline double get_y() const
  {
    return get_decay_time_kernel__X__pre_spikes() - get_rise_time_kernel__X__pre_spikes();
  }


  // -------------------------------------------------------------------------
  //   Function declarations
  // -------------------------------------------------------------------------



  /**
   * Update internal state (``S_``) of the synapse according to the dynamical equations defined in the model and the statements in the ``update`` block.
  **/
  inline void
  update_internal_state_(double t_start, double timestep, const stdp_stp__with_iaf_psc_exp_wtaCommonSynapseProperties& cp);

  void recompute_internal_variables();
  int get_thread();



public:
  // this line determines which common properties to use
  typedef stdp_stp__with_iaf_psc_exp_wtaCommonSynapseProperties CommonPropertiesType;

  typedef Connection< targetidentifierT > ConnectionBase;

  /**
  * Default constructor.
  *
  * Sets default values for all parameters (skipping common properties).
  *
  * Needed by GenericConnectorModel.
  */
  stdp_stp__with_iaf_psc_exp_wta();

  /**
  * Copy constructor from a property object.
  *
  * Sets default values for all parameters (skipping common properties).
  *
  * Needs to be defined properly in order for GenericConnector to work.
  */
  stdp_stp__with_iaf_psc_exp_wta( const stdp_stp__with_iaf_psc_exp_wta& rhs );

  // Explicitly declare all methods inherited from the dependent base
  // ConnectionBase. This avoids explicit name prefixes in all places these
  // functions are used. Since ConnectionBase depends on the template parameter,
  // they are not automatically found in the base class.
  using ConnectionBase::get_delay_steps;
  using ConnectionBase::set_delay_steps;
  using ConnectionBase::get_delay;
  using ConnectionBase::set_delay;
  using ConnectionBase::get_rport;
  using ConnectionBase::get_target;


  class ConnTestDummyNode : public ConnTestDummyNodeBase
  {
  public:
    // Ensure proper overriding of overloaded virtual functions.
    // Return values from functions are ignored.
    using ConnTestDummyNodeBase::handles_test_event;
    port
    handles_test_event( SpikeEvent&, rport )
    {
      return invalid_port;
    }
    port
    handles_test_event( RateEvent&, rport )
    {
      return invalid_port;    }
    port
    handles_test_event( DataLoggingRequest&, rport )
    {
      return invalid_port;    }
    port
    handles_test_event( CurrentEvent&, rport )
    {
      return invalid_port;    }
    port
    handles_test_event( ConductanceEvent&, rport )
    {
      return invalid_port;    }
    port
    handles_test_event( DoubleDataEvent&, rport )
    {
      return invalid_port;    }
    port
    handles_test_event( DSSpikeEvent&, rport )
    {
      return invalid_port;    }
    port
    handles_test_event( DSCurrentEvent&, rport )
    {
      return invalid_port;    }
  };

  /**
   *  special case for weights in NEST: only in case a NESTML state variable was decorated by @nest::weight
  **/
  inline void set_weight(double w)
  {
    // special case for variable marked with @nest::weight decorator
    set_w(w);
    return;

    // no variable was decorated by @nest::weight, so no "weight" defined from the NEST perspective
    assert(0);
  }


/*  inline double get_weight() {
  }
*/
  void
  check_connection( Node& s,
    Node& t,
    rport receptor_type,
    const CommonPropertiesType& cp )
  {
    ConnTestDummyNode dummy_target;
    ConnectionBase::check_connection_( dummy_target, s, t, receptor_type );
    try {
      dynamic_cast<iaf_psc_exp_wta__with_stdp_stp&>(t);
    }
    catch (std::bad_cast &exp) {
      std::cout << "wrong type of neuron connected! Synapse 'stdp_stp__with_iaf_psc_exp_wta' will only work with neuron 'iaf_psc_exp_wta__with_stdp_stp'.\n";
      exit(1);
    }

    t.register_stdp_connection( t_lastspike_ - get_delay(), get_delay() );
  }

  void
  send( Event& e, const thread tid, const stdp_stp__with_iaf_psc_exp_wtaCommonSynapseProperties& cp )
  {
    const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function

    auto get_thread = [tid]()
    {
        return tid;
    };

    // synapse STDP depressing/facilitation dynamics
    const double __t_spike = e.get_stamp().get_ms();
#ifdef DEBUG
    std::cout << "stdp_stp__with_iaf_psc_exp_wta::send(): handling pre spike at t = " << __t_spike << std::endl;
#endif
    // use accessor functions (inherited from Connection< >) to obtain delay and target
    iaf_psc_exp_wta__with_stdp_stp* __target = static_cast<iaf_psc_exp_wta__with_stdp_stp*>(get_target(tid));
    assert(__target != NULL);
    const double __dendritic_delay = get_delay();
    const bool pre_before_post_update = 0;
    bool pre_before_post_flag = false;

    if (t_lastspike_ < 0.)
    {
        // this is the first presynaptic spike to be processed
        t_lastspike_ = 0.;
    }
    double timestep = 0;

    {
      // get spike history in relevant range (t1, t2] from post-synaptic neuron
      std::deque< histentry__iaf_psc_exp_wta__with_stdp_stp >::iterator start;
      std::deque< histentry__iaf_psc_exp_wta__with_stdp_stp >::iterator finish;
      // For a new synapse, t_lastspike_ contains the point in time of the last
      // spike. So we initially read the
      // history(t_last_spike - dendritic_delay, ..., T_spike-dendritic_delay]
      // which increases the access counter for these entries.
      // At registration, all entries' access counters of
      // history[0, ..., t_last_spike - dendritic_delay] have been
      // incremented by Archiving_Node::register_stdp_connection(). See bug #218 for
      // details.
      __target->get_history__( t_lastspike_ - __dendritic_delay,
        __t_spike - __dendritic_delay,
        &start,
        &finish );
      // facilitation due to post-synaptic spikes since last pre-synaptic spike
      while ( start != finish )
      {
        const double minus_dt = t_lastspike_ - ( start->t_ + __dendritic_delay );
        // get_history() should make sure that
        // start->t_ > t_lastspike_ - dendritic_delay, i.e. minus_dt < 0
        assert( minus_dt < -kernel().connection_manager.get_stdp_eps() );

        if (pre_before_post_update && start->t_ == __t_spike - __dendritic_delay)
        {
          pre_before_post_flag = true;
          break;  // this would in any case have been the last post spike to be processed
        }

#ifdef DEBUG
        std::cout << "\tprocessing post spike at t = " << start->t_ << std::endl;
#endif

        /**
         * update synapse internal state from `t_lastspike_` to `start->t_`
        **/

        update_internal_state_(t_lastspike_, (start->t_ + __dendritic_delay) - t_lastspike_, cp);

        timestep += (start->t_ + __dendritic_delay) - t_lastspike_;

        const double _tr_t = start->t_;      
      /**
       *  NESTML generated onReceive code block for postsynaptic port "post_spikes" begins here!
      **//* generated by directives/Block.jinja2 */ /* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Assignment.jinja2 */ 
          S_.w += get_y() * P_.c * std::exp((-(get_w()))) - 1;

        /**
         * internal state has now been fully updated to `start->t_ + __dendritic_delay`
        **/

        t_lastspike_ = start->t_ + __dendritic_delay;
        ++start;
      }
    }

    /**
     * update synapse internal state from `t_lastspike_` to `__t_spike`
    **/

    update_internal_state_(t_lastspike_, __t_spike - t_lastspike_, cp);

    const double _tr_t = __t_spike - __dendritic_delay;

#ifdef DEBUG
    std::cout << "\tDepressing, old w = " << S_.w << "\n";
#endif
    //std::cout << "r2 = " << get_tr_r2() << std::endl;    
    /**
     *  NESTML generated onReceive code block for presynaptic port "pre_spikes" begins here!
    **//* generated by directives/Block.jinja2 */ /* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Assignment.jinja2 */ 
        S_.w += 0;/* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Declaration.jinja2 */ 
    double w_eff = get_u() * get_x() * get_w();/* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Assignment.jinja2 */ 
        S_.x -= get_x() * get_u();/* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Assignment.jinja2 */ 
        S_.u += P_.U * (1 - get_u());/* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/FunctionCall.jinja2 */ 

            set_delay( P_.d );
            const long __delay_steps = nest::Time::delay_ms_to_steps( get_delay() );
            set_delay_steps(__delay_steps);
            e.set_receiver( *__target );
      e.set_weight( w_eff );
      // use accessor functions (inherited from Connection< >) to obtain delay in steps and rport
      e.set_delay_steps( get_delay_steps() );
      e.set_rport( get_rport() );
    e();
    ;

#ifdef DEBUG
    std::cout <<"\t-> new w = " << S_.w << std::endl;
#endif

    /**
     *  update all convolutions with pre spikes
    **/

// XXX: TODO: increment with initial value instead of 1
    S_.decay_time_kernel__X__pre_spikes += 1.;// XXX: TODO: increment with initial value instead of 1
    S_.rise_time_kernel__X__pre_spikes += 1.;

    /**
     *  in case pre and post spike time coincide and pre update takes priority
    **/

    if (pre_before_post_flag)
    {      
      /**
       *  NESTML generated onReceive code block for postsynaptic port "post_spikes" begins here!
      **/
      #ifdef DEBUG
      std::cout << "\tFacilitating from c = " << S_.c << " (using trace = " << S_.pre_tr << ")";
      #endif/* generated by directives/Block.jinja2 */ /* generated by directives/Statement.jinja2 */ /* generated by directives/SmallStatement.jinja2 */ /* generated by directives/Assignment.jinja2 */ 
          S_.w += get_y() * P_.c * std::exp((-(get_w()))) - 1;
#ifdef DEBUG
      std::cout << " to " << S_.c << std::endl;
#endif
    }

    /**
     *  synapse internal state has now been fully updated to `__t_spike`
    **/

    t_lastspike_ = __t_spike;
  }

  void get_status( DictionaryDatum& d ) const;

  void set_status( const DictionaryDatum& d, ConnectorModel& cm );
  nest::normal_distribution normal_dev_; //!< random deviate generator
};


template < typename targetidentifierT >
void
stdp_stp__with_iaf_psc_exp_wta< targetidentifierT >::get_status( DictionaryDatum& __d ) const
{
  ConnectionBase::get_status( __d );
  def< long >( __d, names::size_of, sizeof( *this ) );

  // parameters
  def< double >( __d, names::delay, P_.d );  /* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_U, get_U());  /* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_tau_d, get_tau_d());  /* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_tau_f, get_tau_f());  /* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_tau_rise, get_tau_rise());  /* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_tau_decay, get_tau_decay());  /* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_c, get_c());

  // initial values for state variables in ODE or kernel  /* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_w, get_w());
  // special treatment for variable marked with @nest::name decorator
  def<double>(__d, names::weight, get_w());/* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_u, get_u());/* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_x, get_x());/* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_decay_time_kernel__X__pre_spikes, get_decay_time_kernel__X__pre_spikes());/* generated by directives/WriteInDictionary.jinja2 */ 
  def<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_rise_time_kernel__X__pre_spikes, get_rise_time_kernel__X__pre_spikes());
}

template < typename targetidentifierT >
void
stdp_stp__with_iaf_psc_exp_wta< targetidentifierT >::set_status( const DictionaryDatum& __d,
  ConnectorModel& cm )
{
  // parameters  /* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_d = get_d();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_d, tmp_d);  /* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_U = get_U();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_U, tmp_U);  /* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_tau_d = get_tau_d();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_tau_d, tmp_tau_d);  /* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_tau_f = get_tau_f();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_tau_f, tmp_tau_f);  /* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_tau_rise = get_tau_rise();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_tau_rise, tmp_tau_rise);  /* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_tau_decay = get_tau_decay();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_tau_decay, tmp_tau_decay);  /* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_c = get_c();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_c, tmp_c);

  // initial values for state variables in ODE or kernel  /* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_w = get_w();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_w, tmp_w);
  // special treatment for variables marked with @nest::name decorator
  tmp_w = get_w();
  updateValue<double>(__d, "weight", tmp_w);/* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_u = get_u();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_u, tmp_u);/* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_x = get_x();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_x, tmp_x);/* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_decay_time_kernel__X__pre_spikes = get_decay_time_kernel__X__pre_spikes();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_decay_time_kernel__X__pre_spikes, tmp_decay_time_kernel__X__pre_spikes);/* generated by directives/ReadFromDictionaryToTmp.jinja2 */ 
  double tmp_rise_time_kernel__X__pre_spikes = get_rise_time_kernel__X__pre_spikes();
  updateValue<double>(__d, nest::stdp_stp__with_iaf_psc_exp_wta_names::_rise_time_kernel__X__pre_spikes, tmp_rise_time_kernel__X__pre_spikes);


  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  ConnectionBase::set_status( __d, cm );

  // if we get here, temporaries contain consistent set of properties
  // set parameters  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_d(tmp_d);  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_U(tmp_U);  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_tau_d(tmp_tau_d);  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_tau_f(tmp_tau_f);  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_tau_rise(tmp_tau_rise);  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_tau_decay(tmp_tau_decay);  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_c(tmp_c);

  // set state  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_w(tmp_w);  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_u(tmp_u);  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_x(tmp_x);  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_decay_time_kernel__X__pre_spikes(tmp_decay_time_kernel__X__pre_spikes);  /* generated by directives/AssignTmpDictionaryValue.jinja2 */ 
  set_rise_time_kernel__X__pre_spikes(tmp_rise_time_kernel__X__pre_spikes);

  // check invariants



  // special treatment of NEST delay
  set_delay(
get_d());

  // recompute internal variables in case they are dependent on parameters or state that might have been updated in this call to set_status()
  recompute_internal_variables();
}

/**
 * NESTML internals block symbols initialisation
**/
template < typename targetidentifierT >
void stdp_stp__with_iaf_psc_exp_wta< targetidentifierT >::recompute_internal_variables()
{
  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function


    /* generated by directives/MemberInitialization.jinja2 */ 
    V_.__P__u__u = std::exp((-(V_.__h)) / P_.tau_f); // as real
    /* generated by directives/MemberInitialization.jinja2 */ 
    V_.__P__x__x = std::exp((-(V_.__h)) / P_.tau_d); // as real
    /* generated by directives/MemberInitialization.jinja2 */ 
    V_.__P__decay_time_kernel__X__pre_spikes__decay_time_kernel__X__pre_spikes = std::exp((-(V_.__h)) / P_.tau_decay); // as real
    /* generated by directives/MemberInitialization.jinja2 */ 
    V_.__P__rise_time_kernel__X__pre_spikes__rise_time_kernel__X__pre_spikes = std::exp((-(V_.__h)) / P_.tau_rise); // as real
}
template < typename targetidentifierT >
int stdp_stp__with_iaf_psc_exp_wta< targetidentifierT >::get_thread()
{
    const thread tid = kernel().vp_manager.get_thread_id();
    return tid;
}

/**
 * constructor
**/
template < typename targetidentifierT >
stdp_stp__with_iaf_psc_exp_wta< targetidentifierT >::stdp_stp__with_iaf_psc_exp_wta() : ConnectionBase()
{
  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function
  /* generated by directives/MemberInitialization.jinja2 */ 
  P_.d = 1.0; // as ms
  /* generated by directives/MemberInitialization.jinja2 */ 
  P_.U = ((0.5) + (0.25) * normal_dev_( nest::get_vp_specific_rng( get_thread() ) )); // as real
  /* generated by directives/MemberInitialization.jinja2 */ 
  P_.tau_d = ((110) + (55) * normal_dev_( nest::get_vp_specific_rng( get_thread() ) )); // as ms
  /* generated by directives/MemberInitialization.jinja2 */ 
  P_.tau_f = ((5) + (2.5) * normal_dev_( nest::get_vp_specific_rng( get_thread() ) )); // as ms
  /* generated by directives/MemberInitialization.jinja2 */ 
  P_.tau_rise = 2; // as ms
  /* generated by directives/MemberInitialization.jinja2 */ 
  P_.tau_decay = 20; // as ms
  /* generated by directives/MemberInitialization.jinja2 */ 
  P_.c = 0.05 * std::exp(5); // as real

  V_.__h = nest::Time::get_resolution().get_ms();
  recompute_internal_variables();

  // initial values for state variables in ODE or kernel
  /* generated by directives/MemberInitialization.jinja2 */ 
  S_.w = 0.0; // as real
  /* generated by directives/MemberInitialization.jinja2 */ 
  S_.u = P_.U; // as real
  /* generated by directives/MemberInitialization.jinja2 */ 
  S_.x = 1.0; // as real
  /* generated by directives/MemberInitialization.jinja2 */ 
  S_.decay_time_kernel__X__pre_spikes = 0; // as real
  /* generated by directives/MemberInitialization.jinja2 */ 
  S_.rise_time_kernel__X__pre_spikes = 0; // as real

  t_lastspike_ = 0.;
}

/**
 * copy constructor
**/
template < typename targetidentifierT >
stdp_stp__with_iaf_psc_exp_wta< targetidentifierT >::stdp_stp__with_iaf_psc_exp_wta( const stdp_stp__with_iaf_psc_exp_wta< targetidentifierT >& rhs )
: ConnectionBase( rhs )
{
    P_.d = rhs.P_.d;
    P_.U = rhs.P_.U;
    P_.tau_d = rhs.P_.tau_d;
    P_.tau_f = rhs.P_.tau_f;
    P_.tau_rise = rhs.P_.tau_rise;
    P_.tau_decay = rhs.P_.tau_decay;
    P_.c = rhs.P_.c;

  // state variables in ODE or kernel
    S_.w = rhs.S_.w;
    S_.u = rhs.S_.u;
    S_.x = rhs.S_.x;
    S_.decay_time_kernel__X__pre_spikes = rhs.S_.decay_time_kernel__X__pre_spikes;
    S_.rise_time_kernel__X__pre_spikes = rhs.S_.rise_time_kernel__X__pre_spikes;

    //weight_ = get_named_parameter<double>(names::weight);
    //set_weight( *rhs.weight_ );
    t_lastspike_  = rhs.t_lastspike_;

    // special treatment of NEST delay
    set_delay(rhs.get_delay());
}

template < typename targetidentifierT >
inline void
stdp_stp__with_iaf_psc_exp_wta< targetidentifierT >::update_internal_state_(double t_start, double timestep, const stdp_stp__with_iaf_psc_exp_wtaCommonSynapseProperties& cp)
{
  if (timestep < 1E-12)
  {
#ifdef DEBUG
    std::cout << "\tupdate_internal_state_() called with dt < 1E-12; skipping update\n" ;
#endif
    return;
  }

  const double __resolution = timestep;  // do not remove, this is necessary for the resolution() function

#ifdef DEBUG
  std::cout<< "\tUpdating internal state: t_start = " << t_start << ", dt = " << timestep << "\n";
#endif
  const double old___h = V_.__h;
  V_.__h = timestep;
  recompute_internal_variables();  /* generated by directives/AnalyticIntegrationStep_begin.jinja2 */ 
  double u__tmp = P_.U - V_.__P__u__u * (P_.U - get_u());
  double x__tmp = V_.__P__x__x * (get_x() - 1) + 1;
  double decay_time_kernel__X__pre_spikes__tmp = V_.__P__decay_time_kernel__X__pre_spikes__decay_time_kernel__X__pre_spikes * get_decay_time_kernel__X__pre_spikes();
  double rise_time_kernel__X__pre_spikes__tmp = V_.__P__rise_time_kernel__X__pre_spikes__rise_time_kernel__X__pre_spikes * get_rise_time_kernel__X__pre_spikes();
  /* replace analytically solvable variables with precisely integrated values  *//* generated by directives/AnalyticIntegrationStep_end.jinja2 */ 
  S_.u = u__tmp;
  S_.x = x__tmp;
  S_.decay_time_kernel__X__pre_spikes = decay_time_kernel__X__pre_spikes__tmp;
  S_.rise_time_kernel__X__pre_spikes = rise_time_kernel__X__pre_spikes__tmp;
    V_.__h = old___h;
    recompute_internal_variables();  // XXX: can be skipped?
}

} // namespace

#endif /* #ifndef STDP_STP__WITH_IAF_PSC_EXP_WTA_H */