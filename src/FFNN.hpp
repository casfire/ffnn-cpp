#ifndef _FFNN_HPP_
#define _FFNN_HPP_

#include <vector>
#include <functional>
#include <initializer_list>

namespace FFNN {
	
	template <class T>
	struct TransferFunction {
		std::function<T(const T&)> transfer;
		std::function<T(const T&)> derivative;
		std::function<T(const T&)> inverse;
	};
	
	struct Sigmoid {
		template <class T>
		operator TransferFunction<T>() const; 
	};
	
	struct Heaviside {
		template <class T>
		operator TransferFunction<T>() const;
	};
	
	struct HyperbolicTangent {
		template <class T>
		operator TransferFunction<T>() const;
	};
	
	template <class T, class Container = std::vector<T>>
	class Layer {
	public:
		
		using container_type = Container;
		using size_type = typename Container::size_type;
		
		Layer (
			size_type inputs, size_type outputs,
			TransferFunction<T> transfer = Sigmoid()
		);
		
		size_type  inputs() const { return  in_size; }
		size_type outputs() const { return out_size; }
		size_type weights() const { return w.size(); }
		const T& weight(size_type i) const;
		      T& weight(size_type i);
		const T& weight(size_type in, size_type out) const;
		      T& weight(size_type in, size_type out);
		const T&   bias(size_type out) const;
		      T&   bias(size_type out);
		
		template <
			class InputIter, class OutputIter
		> void compute (
			InputIter input, OutputIter output
		) const;
		
		template <
			class InputIter, class OutputIter,
			class TargetIter, class BackTargetIter
		> void train (
			const T &rate,
			InputIter input, OutputIter output,
			TargetIter target, BackTargetIter back
		);
		
	private:
		
		size_type in_size, out_size;
		TransferFunction<T> tf;
		Container w;
		
	};
	
	template <class T, class C = std::vector<T>>
	struct ComputeBuffer {
		C A, B;
		typename C::iterator beginA(typename C::size_type s);
		typename C::iterator beginB(typename C::size_type s);
	};
	
	template <
		class T, class B = ComputeBuffer<T>,
		class C = std::vector<B>
	> struct TrainBuffer : public C {};
	
	template <
		class T, class LayerT = Layer<T>,
		class Container = std::vector<LayerT>
	> class Network {
	public:
		
		using container_type  = Container;
		using layer_type      = LayerT;
		using size_type       = typename Container::size_type;
		using layer_size_type = typename LayerT::size_type;
		
		Network (
			layer_size_type inputs,
			std::initializer_list<layer_size_type> nodes,
			TransferFunction<T> transfer = Sigmoid()
		);
		
		layer_size_type  inputs() const { return l.front().inputs(); }
		layer_size_type outputs() const { return l.back().outputs(); }
		size_type        layers() const { return l.size();           }
		
		const LayerT& layer(size_type i) const { return *std::next(l.cbegin(), i); }
		      LayerT& layer(size_type i)       { return *std::next(l.begin(),  i); }
		
		template <class InputIter, class OutputIter>
		void compute (
			InputIter input, OutputIter output,
			ComputeBuffer<T> &buffer
		) const;
		
		template <class InputIter, class TargetIter>
		void train (
			const T &rate,
			InputIter input, TargetIter target,
			TrainBuffer<T> &buffer
		);
		
		template <class InputIter, class OutputIter>
		void compute (
			InputIter input, OutputIter output
		) const;
		
		template <class InputIter, class TargetIter>
		void train (
			const T &rate,
			InputIter input, TargetIter target
		);
		
	private:
		
		Container l;
		
	};
	
}

#include "FFNN.inl"

#endif
