(() => {
  document.documentElement.classList.add('has-js');

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const progress = document.createElement('div');
  progress.className = 'scroll-progress';
  progress.setAttribute('aria-hidden', 'true');
  document.body.prepend(progress);

  const updateProgress = () => {
    const available = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = available > 0 ? Math.min(1, Math.max(0, window.scrollY / available)) : 0;
    progress.style.transform = `scaleX(${ratio})`;
  };

  updateProgress();
  window.addEventListener('scroll', updateProgress, { passive: true });
  window.addEventListener('resize', updateProgress, { passive: true });

  const targets = [...document.querySelectorAll(
    '.welcome, .section-block, .references, .about-card'
  )];

  targets.forEach((target, index) => {
    target.classList.add('reveal-ready');
    target.style.setProperty('--reveal-delay', `${Math.min(index, 5) * 45}ms`);
  });

  if (reducedMotion || !('IntersectionObserver' in window)) {
    targets.forEach(target => target.classList.add('is-visible'));
    return;
  }

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });

  targets.forEach(target => observer.observe(target));
})();
