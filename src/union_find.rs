use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

/// UnionFind pointer.
#[derive(Default)]
pub struct UnionFindRef<T>(Rc<RefCell<UnionFindNode<T>>>);

#[derive(Clone, PartialEq)]
enum UnionFindNode<T> {
    Final(T),
    Follow(UnionFindRef<T>),
}
impl<T: Default> Default for UnionFindNode<T> {
    fn default() -> Self {
        Self::Final(T::default())
    }
}
impl<T: std::fmt::Debug> std::fmt::Debug for UnionFindNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Final(t) => t.fmt(f),
            Self::Follow(r) => r.fmt(f),
        }
    }
}
impl<T: std::fmt::Debug> std::fmt::Debug for UnionFindRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.borrow().fmt(f)
    }
}
impl<T: PartialEq> PartialEq for UnionFindRef<T> {
    fn eq(&self, other: &Self) -> bool {
        let mut left = self.clone();
        let mut right = other.clone();
        let left_inner = left.inner();
        let right_inner = right.inner();
        *left_inner == *right_inner
    }
}
impl<T> Clone for UnionFindRef<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Clone> From<T> for UnionFindRef<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}
impl<T> UnionFindRef<T> {
    pub fn new(t: T) -> Self {
        Self(Rc::new(RefCell::new(UnionFindNode::Final(t))))
    }
    pub fn ptr_eq(&self, other: &Self) -> bool {
        let mut left = self.clone();
        let mut right = other.clone();
        left.compress();
        right.compress();
        Rc::ptr_eq(&left.0, &right.0)
    }

    fn end(&self) -> Self {
        let mut end = self.clone();
        loop {
            let next = match &*end.0.borrow() {
                UnionFindNode::Follow(other) => other.clone(),
                UnionFindNode::Final(_) => break,
            };
            end = next;
        }
        end
    }
    fn compress(&mut self) -> &Self {
        if matches!(&*self.0.borrow(), UnionFindNode::Final(_)) {
            return self;
        }
        // Follow path to end.
        let end = self.end();

        // Compress the path.
        let mut current = self.clone();
        loop {
            let next = match &*end.0.borrow() {
                UnionFindNode::Follow(other) => other.clone(),
                UnionFindNode::Final(_) => break,
            };
            *current.0.borrow_mut() = UnionFindNode::Follow(end.clone());
            current = next;
        }
        *self = end;
        self
    }
    /// Makes this union-find node an alias of `other`.
    pub fn follow(&mut self, other: &UnionFindRef<T>) {
        *self.0.borrow_mut() = UnionFindNode::Follow(other.clone());
        *self = other.clone();
    }
    /// Immutable borrow of the inner T.
    pub fn inner(&mut self) -> Ref<T> {
        self.compress();
        Ref::map(self.0.borrow(), |node| {
            if let UnionFindNode::Final(t) = node {
                t
            } else {
                unreachable!()
            }
        })
    }
    /// Mutable borrow of the inner T.
    pub fn inner_mut(&mut self) -> RefMut<T> {
        self.compress();
        RefMut::map(self.0.borrow_mut(), |node| {
            if let UnionFindNode::Final(t) = node {
                t
            } else {
                unreachable!()
            }
        })
    }
}
impl<T: Clone> UnionFindRef<T> {
    pub fn clone_inner(&self) -> T {
        self.clone().inner().clone()
    }
}
