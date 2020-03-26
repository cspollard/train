{-# LANGUAGE TupleSections #-}
{-# LANGUAGE DerivingVia #-}

module Train where

import Prelude hiding ((<*>), (<+>), const, id)


-- goal: represent category of differentiable functions
-- and the Diff functor


-- this only works for first derivative...
-- i.e. taking the derivative twice will do something odd.
newtype D a b = D { runD :: a -> (b, D b a) }


appD :: D a b -> a -> b
appD (D f) = fst . f


gradD :: D a b -> (a, b) -> a
gradD (D f) (a, b) = flip appD b . snd $ f a


backpropD :: ((b, b) -> b) -> D a b -> (a, b) -> (b, D a b)
backpropD loss (D f) (a, targ) =
  let (pred, D f') = f a
  in (pred, snd . f' $ loss (targ, pred))


trainD :: ((b, b) -> b) -> D a b -> [(a, b)] -> D a b
trainD loss = foldr (\ins d -> snd $ backpropD loss d ins)


(<+>) :: Semigroup a => a -> a -> a
(<+>) = (<>)


class Monoid a => Group a where
  inv :: a -> a


class Group a => Rng a where
  (<*>) :: a -> a -> a


newtype WN a = WN a deriving (Eq, Ord, Num, Fractional, Floating, Enum, Show) via a


instance Num a => Semigroup (WN a) where
  (<>) = (+)


instance Num a => Monoid (WN a) where 
  mempty = 0

instance Num a => Group (WN a) where
  inv = negate


instance Num a => Rng (WN a) where
  (<*>) = (*)
  

id :: D a a
id = D $ \a -> (a, id)


comp :: (D a b, D b c) -> D a c
comp (D f, D g) = D go
  where
    go a =
      let (b, upf) = f a
          (c, upg) = g b
      in (c, comp (upg, upf))


infixr 1 >>>
infixr 1 <<<
(>>>) = curry comp
(<<<) = flip (>>>)


prod :: (D a b, D c d) -> D (a, c) (b, d)
prod (D f, D g) = D $ \(a, c) ->
  let (b, f') = f a
      (d, g') = g c
  in ((b, d), prod (f', g'))


infixr 3 ***
(***) = curry prod


first d = d *** id
second d = id *** d


dup :: Semigroup a => D a (a, a)
dup = D $ \a -> ((a, a), plus)


plus :: Semigroup a => D (a, a) a
plus = D $ \(a, b) -> (a <+> b, dup)


neg :: Group a => D a a
neg = D $ \a -> (inv a, neg)


minus :: Group a => D (a, a) a
minus = second neg >>> plus


times :: Rng a => D (a, a) a
times = D $ \(a, b) -> (a <*> b, D $ \dc -> ((a <*> dc, b <*> dc), times))


fork :: Semigroup a => (D a b, D a c) -> D a (b, c)
fork (f, g) = comp (dup, prod (f, g))


const :: Semigroup a => a -> D () a
const a = D $ \() -> (a, D $ \da -> ((), const $ a <+> da))


param :: Semigroup a => a -> D b c -> D b (a, c)
param a f = f >>> inr >>> first (const a)


swap :: D (a, b) (b, a)
swap = D $ \(a, b) -> ((b, a), D $ \(b, a) -> ((a, b), swap))


inl :: D a (a, ())
inl = D $ \a -> ((a, ()), D $ \(a, ()) -> (a, inl))


inr :: D a ((), a)
inr = comp (inl, swap)


exl :: Monoid b => D (a, b) a
exl = D $ \(a, _) -> (a, D $ \da -> ((da, mempty), exl))


exr :: Monoid a => D (a, b) b
exr = swap >>> exl

linear :: Rng a => a -> a -> D a a
linear m b = slope m >>> bias b


bias :: Semigroup a => a -> D a a
bias b = inl >>> second (const b) >>> plus


slope :: Rng a => a -> D a a
slope m = inl >>> second (const m) >>> times


sqr :: Rng a => D a a
sqr = dup >>> times


-- maximize - (targ - pred)**2
-- gradient: 2 * (targ - pred)
magSqrLoss :: Num a => (a, a) -> a
magSqrLoss (targ, pred) = 2 * (targ - pred)
