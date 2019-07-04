/*
 * MinerUtils.h
 *
 * Copyright (C) 2018 SingularityNET Foundation
 *
 * Author: Nil Geisweiller
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */
#ifndef OPENCOG_MINER_UTILS_H_
#define OPENCOG_MINER_UTILS_H_

#include <opencog/atoms/base/Handle.h>

#include "Valuations.h"

namespace opencog
{

/**
 * Collection of static methods for the pattern miner.
 */
class MinerUtils
{
public:
	/**
	 * Wrap conjuncts (including unary) with PresentLink rather than
	 * AndLink.
	 */
	static const bool use_present_link = true;

	/**
	 * Given valuations produce all shallow abstractions reaching
	 * minimum support, over all variables. It basically applies
	 * focus_shallow_abstract recursively. See the specification of
	 * focus_shallow_abstract for more information.
	 *
	 * For instance, given
	 *
	 * valuations =
	 *   { { X->(Inheritance (Concept "A") (Concept "B")), Y->(Concept "C") },
	 *     { X->(Inheritance (Concept "B") (Concept "C")), Y->(Concept "D") },
	 *     { X->(Concept "E"), Y->(Concept "D") } }
	 * ms = 2
	 *
	 * shallow_abstract(valuations) =
	 *  {
	 *    ;; Shallow abstractions of X
	 *    { (Lambda
	 *        (VariableList
	 *          (Variable "$X1")
	 *          (Variable "$X2"))
	 *        (Inheritance
	 *          (Variable "$X1")
	 *          (Variable "$X2"))) },
	 *    ;; Shallow abstractions of Y
	 *    { (Concept "D") }
	 *  }
	 */
	static HandleSetSeq shallow_abstract(const Valuations& valuations, unsigned ms);

	/**
	 * Given valuations produce all shallow abstractions reaching
	 * minimum support based on the values associated to the variable
	 * in focus. This shallow abstractions include
	 *
	 * 1. Single operator patterns, like (Lambda X Y (Inheritance X Y))
	 * 2. Constant nodes, like (Concept "A")
	 * 3. Remain variable after the front one, x2, ..., xn
	 *
	 * Composing these 3 sorts of abstractions are enough to generate
	 * all possible patterns.
	 *
	 * For instance, given
	 *
	 * valuations =
	 *   { { X->(Inheritance (Concept "A") (Concept "B")), Y->(Concept "C") },
	 *     { X->(Inheritance (Concept "B") (Concept "C")), Y->(Concept "D") },
	 *     { X->(Concept "E"), Y->(Concept "D") } }
	 * ms = 2
	 *
	 * front_shallow_abstract(valuations) = { (Lambda
	 *                                          (VariableList
	 *                                            (Variable "$X1")
	 *                                            (Variable "$X2"))
	 *                                          (Inheritance
	 *                                            (Variable "$X1")
	 *                                            (Variable "$X2"))) }
	 */
	static HandleSet focus_shallow_abstract(const Valuations& valuations, unsigned ms);

	/**
	 * Return true iff h is a node or a nullary link.
	 */
	static bool is_nullary(const Handle& h);

	/**
	 * Given an atom, a value, return its corresponding shallow
	 * abstraction. A shallow abstraction of an atom is
	 *
	 * 1. itself if it is nullary (see is_nullary)
	 *
	 * 2. (Lambda (VariableList X1 ... Xn) (L X1 ... Xn) if it is a
	 *    link of arity n.
	 *
	 * For instance, with
	 *
	 * text = (Inheritance (Concept "a") (Concept "b"))
	 *
	 * shallow_patterns(text) = (Lambda
	 *                            (VariableList
	 *                              (Variable "$X1")
	 *                              (Variable "$X2"))
	 *                            (Inheritance
	 *                              (Variable "$X1")
	 *                              (Variable "$X2")))
	 *
	 * TODO: we may want to support types in variable declaration.
	 */
	static Handle shallow_abstract_of_val(const Handle& value);

	/**
	 * Wrap a VariableList around a variable list, if more than one
	 * variable, otherwise return that one variable.
	 */
	static Handle variable_list(const HandleSeq& vars);

	/**
	 * Wrap a LambdaLink around a vardecl and body.
	 */
	static Handle lambda(const Handle& vardecl, const Handle& body);

	/**
	 * Wrap a QuoteLink around h
	 */
	static Handle quote(const Handle& h);

	/**
	 * Wrap a UnquoteLink around h
	 */
	static Handle unquote(const Handle& h);

	/**
	 * Wrap a LocalQuote link around h (typically used if it is a link
	 * of type AndLink. That is in order not to produce
	 * multi-conjuncts patterns when in fact we want to match an
	 * AndLink text.)
	 */
	static Handle local_quote(const Handle& h);

	/**
	 * Given a pattern, and mapping from variables to sub-patterns,
	 * compose (as in function composition) the pattern with the
	 * sub-patterns. That is replace variables in the pattern by their
	 * associated sub-patterns, properly updating the variable
	 * declaration.
	 */
	static Handle compose(const Handle& pattern, const HandleMap& var2pat);

	/**
	 * TODO replace by RewriteLink::beta_reduce
	 *
	 * Given a variable declaration, and a mapping from variables to
	 * variable declaration, produce a new variable declaration, as
	 * obtained by compositing the pattern with the sub-patterns.
	 *
	 * If a variable in vardecl is missing in var2vardecl, then
	 * vardecl is untouched. But if a variable maps to the undefined
	 * Handle, then it is removed from the resulting variable
	 * declaration. That happens in cases where the variable maps to a
	 * constant pattern, i.e. a value. In such case composition
	 * amounts to application.
	 */
	static Handle vardecl_compose(const Handle& vardecl,
	                              const HandleMap& var2subdecl);

	/**
	 * Given a texts concept node, retrieve all its members
	 */
	static HandleSeq get_texts(const Handle& texts_cpt);

	/**
	 * Return the non-negative integer held by a number node.
	 */
	static unsigned get_uint(const Handle& h);

	/**
	 * Given a pattern and a text corpus, calculate the pattern
	 * frequency up to ms (to avoid unnecessary calculations).
	 */
	static unsigned support(const Handle& pattern,
	                        const HandleSeq& texts,
	                        unsigned ms);

	/**
	 * Like support but assumes that pattern is strongly connected (all
	 * its variables depends on other clauses).
	 */
	static unsigned component_support(const Handle& pattern,
	                                  const HandleSeq& texts,
	                                  unsigned ms);

	/**
	 * Calculate if the pattern has enough support w.r.t. to the given
	 * texts, that is whether its frequency is greater than or equal
	 * to ms.
	 */
	static bool enough_support(const Handle& pattern,
	                           const HandleSeq& texts,
	                           unsigned ms);

	/**
	 * Like shallow_abstract(const Valuations&, unsigned) but takes a pattern
	 * and a texts instead, and generate the valuations of the pattern
	 * prior to calling shallow_abstract on its valuations.
	 *
	 * See comment on shallow_abstract(const Valuations&, unsigned) for more
	 * details.
	 */
	static HandleSetSeq shallow_abstract(const Handle& pattern,
	                                     const HandleSeq& texts,
	                                     unsigned ms);

	/**
	 * Return all shallow specializations of pattern with support ms
	 * according to texts.
	 *
	 * mv is the maximum number of variables allowed in the resulting
	 * patterns.
	 */
	static HandleSet shallow_specialize(const Handle& pattern,
	                                    const HandleSeq& texts,
	                                    unsigned ms,
	                                    unsigned mv=UINT_MAX);

	/**
	 * Create a pattern body from clauses, introducing an AndLink if
	 * necessary.
	 */
	static Handle mk_body(const HandleSeq& clauses);

	/**
	 * Given a sequence of clause create a LambdaLink of it without
	 * variable declaration, introducing an AndLink if necessary.
	 */
	static Handle mk_pattern_no_vardecl(const HandleSeq& clauses);

	/**
	 * Given a vardecl and a sequence of clauses, filter the vardecl to
	 * contain only variable of the body, and create a Lambda with
	 * them.
	 */
	static Handle mk_pattern_filtering_vardecl(const Handle& vardecl,
	                                           const HandleSeq& clauses);

	/**
	 * Given a vardecl and a sequence of clauses, build a pattern. If
	 * use_present_link is true, then the result will be
	 *
	 * (Lambda <vardecl> (Present <clauses-1> ... <clauses-n>))
	 */
	static Handle mk_pattern(const Handle& vardecl, const HandleSeq& clauses);

	/**
	 * Given a pattern, split it into smaller patterns of strongly
	 * connected components.
	 */
	static HandleSeq get_component_patterns(const Handle& pattern);

	/**
	 * Like above but consider a sequence of clauses instead of a
	 * pattern, and return a sequence of sequences of clauses.
	 */
	static HandleSeqSeq get_components(const HandleSeq& clauses);

	/**
	 * Given a pattern, split it into its disjuncts.
	 */
	static HandleSeq get_conjuncts(const Handle& pattern);

	/**
	 * Given a pattern and texts, return the satisfying set of the
	 * pattern over the text.
	 *
	 * TODO: ignore permutations for unordered links.
	 *
	 * TODO: ignore duplicates within the same text. For instance if
	 * the pattern is
	 *
	 * (Lambda (LocalQuote (And (Variable "$X") (Variable "$Y"))))
	 *
	 * and the texts is
	 *
	 * { (And (Concept "A") (And (Concept "B") (Concept "C"))) }
	 *
	 * then the result will include 2 results
	 *
	 * { (And (Concept "A") (And (Concept "B") (Concept "C"))),
	 *   (And (Concept "B") (Concept "C")) }
	 *
	 * instead of one
	 *
	 * { (And (Concept "A") (And (Concept "B") (Concept "C"))) }
	 *
	 * Also, the pattern may match any subhypergraph of texts, not just
	 * the root atoms (TODO: we probably don't want that!!!).
	 */
	static Handle restricted_satisfying_set(const Handle& pattern,
	                                        const HandleSeq& texts,
	                                        unsigned ms=UINT_MAX);

	/**
	 * Return true iff the pattern is totally abstract like
	 *
	 * (Lambda
	 *   (Variable "$X")
	 *   (Variable "$X"))
	 *
	 * for a single conjunct. Or
	 *
	 * (Lambda
	 *   (List
	 *     (Variable "$X")
	 *     (Variable "$Y"))
	 *   (And
	 *     (Variable "$X")
	 *     (Variable "$Y"))
	 *
	 * for 2 conjuncts, etc.
	 */
	static bool totally_abstract(const Handle& pattern);

	/**
	 * Generate a list of hopefully unique random variables
	 */
	static HandleSeq gen_rand_variables(size_t n);
	static Handle gen_rand_variable();

	/**
	 * Given a pattern return its variables. If the pattern is not a
	 * scope link (i.e. a constant/text), then return the empty
	 * Variables.
	 */
	static const Variables& get_variables(const Handle& pattern);

	/**
	 * Given a pattern, return its vardecl. If the pattern is not a
	 * scope link (i.e. a constant/text), then return the empty
	 * vardecl.
	 */
	static Handle get_vardecl(const Handle& pattern);

	/**
	 * Given a pattern, return its body. If the pattern is not a scope
	 * link (i.e. a constant/text), then return pattern itself.
	 */
	static const Handle& get_body(const Handle& pattern);

	/**
	 * Given a pattern, return its clause. If the pattern is not a
	 * scope link (i.e. a constant/text), then behavior is undefined.
	 */
	static HandleSeq get_clauses(const Handle& pattern);
	static HandleSeq get_clauses_of_body(const Handle& body);

	/**
	 * Return the number of conjuncts in a pattern. That is, if the
	 * pattern body is an AndLink, then returns its arity, otherwise
	 * if the body is not an AndLink, then return 1, and if it's not a
	 * pattern at all (i.e. not a LambdaLink), then return 0.
	 */
	static unsigned n_conjuncts(const Handle& pattern);

	/**
	 * Remove useless clauses from a body pattern. Useless clauses are
	 * constant clauses, as well as variables that already occur
	 * within an existing clause.
	 */
	static Handle remove_useless_clauses(const Handle& pattern);
	static Handle remove_useless_clauses(const Handle& vardecl,
	                                     const Handle& body);
	static void remove_useless_clauses(const Handle& vardecl,
	                                   HandleSeq& clauses);

	/**
	 * Remove any closes clause (regardless of whether they are
	 * evaluatable or not).
	 */
	static void remove_constant_clauses(const Handle& vardecl,
	                                    HandleSeq& clauses);

	/**
	 * Remove redundant subclauses, such as ones identical to clauses
	 * of there subtrees.
	 */
	static void remove_redundant_subclauses(HandleSeq& clauses);

	/**
	 * Remove redundant clauses.
	 */
	static void remove_redundant_clauses(HandleSeq& clauses);

	/**
	 * Alpha convert pattern so that none of its variables collide with
	 * the variables in other_vars.
	 */
	static Handle alpha_convert(const Handle& pattern,
	                            const Variables& other_vars);

	/**
	 * Construct the conjunction of 2 patterns. If cnjtion is a
	 * conjunction, then expand it with pattern (performing
	 * alpha-conversion when necessary). It is assumed that pattern
	 * cannot be a conjunction itself.
	 *
	 * This method will not attempt to connect the 2 patterns, thus,
	 * assuming that cnjtion is itself strongly connected, the result
	 * will be 2 strongly connected components.
	 */
	static Handle expand_conjunction_disconnect(const Handle& cnjtion,
	                                            const Handle& pattern);

	/**
	 * Like expand_conjunction_disconnect but produced a single
	 * strongly connected component, assuming that cnjtion is itself
	 * strongly connected, given 2 connecting variables, one from
	 * cnjtion, one from pattern.
	 *
	 * Unlike expand_conjunction_disconnect, no alpha conversion is
	 * performed, cnjtion is assumed not to collide with pattern.
	 */
	static Handle expand_conjunction_connect(const Handle& cnjtion,
	                                         const Handle& pattern,
	                                         const Handle& cnjtion_var,
	                                         const Handle& pattern_var);

	/**
	 * Like expand_conjunction_connect but consider a mapping from
	 * variables of pattern to variables of cnjtion.
	 */
	static Handle expand_conjunction_connect(const Handle& cnjtion,
	                                         const Handle& pattern,
	                                         const HandleMap& pv2cv);

	/**
	 * Like expand_conjunction_connect above but recursively consider
	 * all variable mappings from pattern to cnjtion.
	 *
	 * pvi is the variable index of pattern variable declaration.
	 */
	static HandleSet expand_conjunction_connect_rec(const Handle& cnjtion,
	                                                const Handle& pattern,
	                                                const HandleSeq& texts,
	                                                unsigned ms,
	                                                unsigned mv,
	                                                const HandleMap& pv2cv=HandleMap(),
	                                                unsigned pvi=0);

	/**
	 * Given cnjtion and pattern, consider all possible connections
	 * (a.k.a linkages) and expand cnjtion accordingly. For instance if
	 *
	 * cnjtion = (Inheritance X Y)
	 * pattern = (Inheritance Z W)
	 *
	 * return
	 *
	 *   (And (Inheritance X Y) (Inheritance X W))
	 *   (And (Inheritance X Y) (Inheritance Z X))
	 *   (And (Inheritance X Y) (Inheritance Y W))
	 *   (And (Inheritance X Y) (Inheritance X Y))
	 *
	 * It will also only include patterns with minimum support ms
	 * according to texts, and perform alpha-conversion when necessary.
	 * If an expansion is cnjtion itself it will be dismissed.
	 *
	 * mv is the maximum number of variables allowed in the resulting
	 * patterns.
	 */
	static HandleSet expand_conjunction(const Handle& cnjtion,
	                                    const Handle& pattern,
	                                    const HandleSeq& texts,
	                                    unsigned ms,
	                                    unsigned mv=UINT_MAX);

	/**
	 * Return an atom to serve as key to store the support value.
	 */
	static const Handle& support_key();

	/**
	 * Attach the support of a pattern to support_key(). The support is
	 * encoded as double because it is stored as a FloatValue, and its
	 * subsequent processing (probability estimate, etc) requires a
	 * double anyway.
	 */
	static void set_support(const Handle& pattern, double support);

	/**
	 * Get the support of a pattern stored as associated value to
	 * support_key(). If no such value exist then return -1.0.
	 */
	static double get_support(const Handle& pattern);

	/**
	 * Like get_support, but if there is no value associated to
	 * support_key() then calculate and set the support.
	 *
	 * Warning: note that the support is gonna be up to ms, so such
	 * memoization should not be used if ms is to be changed.
	 */
	static double support_mem(const Handle& pattern,
	                          const HandleSeq& texts,
	                          unsigned ms);
};

} // ~namespace opencog

#endif /* OPENCOG_MINER_UTILS_H_ */
