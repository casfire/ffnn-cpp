#include <cmath>

namespace FFNN {
	
	template <class T>
	Sigmoid::operator TransferFunction<T>() const {
		return {
		[](const T& x) -> T { return 1 / (1 + std::exp(-x)); },
		[](const T& y) -> T { return y * (1 - y);            },
		[](const T& y) -> T { return std::log(-y / (y - 1)); }
		};
	}
	
	template <class T>
	Heaviside::operator TransferFunction<T>() const {
		return {
		[](const T& x) -> T { return x >= 0 ? 1 : 0; },
		[](const T&)   -> T { return 1;              },
		std::function<T(const T&)>()
		};
	}
	
	template <class T>
	HyperbolicTangent::operator TransferFunction<T>() const {
		return {
		[](const T& x) -> T { return std::tanh(x);       },
		[](const T& y) -> T { return 1 - std::pow(y, 2); },
		[](const T& y) -> T { return std::atanh(y);      }
		};
	}
	
	template <class T, class Container>
	Layer<T, Container>::Layer (
		size_type inputs, size_type outputs,
		TransferFunction<T> transfer
	) : in_size(inputs), out_size(outputs), tf(transfer),
		w((inputs + 1) * outputs)
	{}
	
	template <class T, class Container>
	const T& Layer<T, Container>::weight (
		size_type in, size_type out
	) const {
		return *std::next(w.cbegin(), out * (1 + in_size) + in + 1);
	}
	
	template <class T, class Container>
	T& Layer<T, Container>::weight (
		size_type in, size_type out
	) {
		return *std::next(w.begin(), out * (1 + in_size) + in + 1);
	}
	
	template <class T, class Container>
	const T& Layer<T, Container>::weight(
		size_type i
	) const {
		return *std::next(w.cbegin(), i);
	}
	
	template <class T, class Container>
	T& Layer<T, Container>::weight(
		size_type i
	) {
		return *std::next(w.begin(), i);
	}
	
	template <class T, class Container>
	const T& Layer<T, Container>::bias (
		size_type out
	) const {
		return *std::next(w.cbegin(), out * (1 + in_size));
	}
	
	template <class T, class Container>
	T& Layer<T, Container>::bias (
		size_type out
	) {
		return *std::next(w.begin(), out * (1 + in_size));
	}
	
	template <class T, class Container>
	template <class InputIter, class OutputIter>
	void Layer<T, Container>::compute (
		InputIter input, OutputIter output
	) const {
		typename Container::const_iterator w_iter = w.cbegin();
		for (size_type out_i = 0; out_i < out_size; out_i++) {
			InputIter in_iter = input;
			T v = *w_iter++;
			for (size_type in_i = 0; in_i < in_size; in_i++)
				v += (*in_iter++) * (*w_iter++);
			*output++ = tf.transfer(v);
		}
	}
	
	template <class T, class Container>
	template <
		class InputIter, class OutputIter,
		class TargetIter, class BackTargetIter
	> void Layer<T, Container>::train (
		const T &rate, InputIter input, OutputIter output,
		TargetIter target, BackTargetIter back
	) {
		typename Container::iterator w_iter = w.begin();
		BackTargetIter back_iter = back;
		InputIter in_iter = input;
		for (size_type in_i = 0; in_i < in_size; in_i++)
			*back_iter++ = *in_iter++;
		for (size_type out_i = 0; out_i < out_size; out_i++) {
			T delta = tf.derivative(*output) * (*output - *target++);
			output++;
			back_iter = back;
			in_iter = input;
			*w_iter++ -= rate * delta;
			for (size_type in_i = 0; in_i < in_size; in_i++) {
				*w_iter -= rate * delta * (*in_iter++);
				*back_iter++ -= delta * (*w_iter++);
			}
		}
	}
	
	template <class T, class LayerT, class Container>
	Network<T, LayerT, Container>::Network (
		layer_size_type inputs,
		std::initializer_list<layer_size_type> nodes,
		TransferFunction<T> transfer
	) {
		layer_size_type prev = inputs;
		for (layer_size_type size : nodes) {
			l.emplace_back(prev, size, transfer);
			prev = size;
		}
	}
	
	template <class T, class C>
	typename C::iterator ComputeBuffer<T, C>::beginA (
		typename C::size_type s 
	) {
		if (A.size() < s) A.resize(s);
		return A.begin();
	}
	
	template <class T, class C>
	typename C::iterator ComputeBuffer<T, C>::beginB (
		typename C::size_type s
	) {
		if (B.size() < s) B.resize(s);
		return B.begin();
	}
	
	template <class T, class LayerT, class Container>
	template <class InputIter, class OutputIter>
	void Network<T, LayerT, Container>::compute (
		InputIter input, OutputIter output,
		ComputeBuffer<T> &buffer
	) const {
		typename Container::const_iterator l_iter = l.cbegin();
		typename Container::const_iterator l_last = --l.cend();
		if (l_iter == l_last) {
			l_iter->compute(input, output);
			return;
		}
		l_iter->compute(input, buffer.beginA(l_iter->outputs()));
		typename LayerT::container_type::iterator b_iter = buffer.A.begin();
		while (++l_iter != l_last) {
			l_iter->compute(b_iter, buffer.beginB(l_iter->outputs()));
			b_iter = buffer.B.begin();
			if (++l_iter == l_last) break;
			l_iter->compute(b_iter, buffer.beginA(l_iter->outputs()));
			b_iter = buffer.A.begin();
		}
		l_iter->compute(b_iter, output);
	}
	
	template <class T, class LayerT, class Container>
	template <class InputIter, class TargetIter>
	void Network<T, LayerT, Container>::train (
		const T &rate,
		InputIter input, TargetIter target,
		TrainBuffer<T> &buffer
	) {
		typename Container::iterator l_iter = l.begin();
		typename Container::iterator l_last = l.end();
		if (buffer.size() < l.size()) buffer.resize(l.size());
		typename TrainBuffer<T>::iterator b_iter = buffer.begin();
		l_iter->compute(input, b_iter->beginA(l_iter->outputs()));
		while (++l_iter != l_last) {
			l_iter->compute(b_iter->A.begin(), std::next(b_iter)->beginA(l_iter->outputs()));
			++b_iter;
		}
		--l_iter;
		l_last = l.begin();
		if (l_iter == l_last) {
			l_iter->train (rate,
				input,  b_iter->A.begin(),
				target, b_iter->beginB(l_iter->inputs())
			);
			return;
		}
		l_iter->train (rate,
			std::prev(b_iter)->A.begin(), b_iter->A.begin(),
			target, b_iter->beginB(l_iter->inputs())
		);
		--b_iter;
		while (--l_iter != l_iter) {
			l_iter->train (rate,
				std::prev(b_iter)->A.begin(), b_iter->A.begin(),
				std::next(b_iter)->B.begin(), b_iter->beginB(l_iter->inputs())
			);
			--b_iter;
		}
		l_iter->train (rate,
			input, b_iter->A.begin(),
			std::next(b_iter)->B.begin(), b_iter->beginB(l_iter->inputs())
		);
	}
	
	template <class T, class LayerT, class Container>
	template <class InputIter, class OutputIter>
	void Network<T, LayerT, Container>::compute (
		InputIter input, OutputIter output
	) const {
		ComputeBuffer<T> temp;
		compute(input, output, temp);
	}
	
	template <class T, class LayerT, class Container>
	template <class InputIter, class TargetIter>
	void Network<T, LayerT, Container>::train (
		const T &rate, InputIter input, TargetIter target
	) {
		TrainBuffer<T> temp;
		train(rate, input, target, temp);
	}
	
}
