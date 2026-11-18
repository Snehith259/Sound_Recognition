/* ═══════════════════════════════════════════════════════════════════════
   SOUNDSENSE · fx.js — ADDITIVE VISUAL EFFECTS & AMBIENT LAYER (v2.0)
   ─────────────────────────────────────────────────────────────────────────
   Contract with app.js:
   · NEVER writes to app-owned nodes except two guarded exceptions:
       1. inline CSS custom props (--mag-x/--mag-y, --gx/--gy, --fx-*)
       2. #conf-pct count-up, which ABORTS instantly if app.js writes to it
   · NEVER removes/renames classes; only adds fx-* classes & html.fx-on
   · Reads app state passively: .recording, .audio-playing, [hidden],
     #mic-level-fill inline width, --ambient-energy
   · Self-disables on touch / coarse pointers and prefers-reduced-motion
   · One shared rAF ticker · transform/opacity writes only · IO lazy-init
   · Pauses when document.hidden; tears down on runtime preference change
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  /* ── 0 · Environment guards ─────────────────────────────────────────── */
  var reducedMQ = window.matchMedia('(prefers-reduced-motion: reduce)');
  var fineMQ    = window.matchMedia('(pointer: fine)');
  var hoverMQ   = window.matchMedia('(hover: hover)');

  if (reducedMQ.matches || !fineMQ.matches || !hoverMQ.matches) {
    document.documentElement.classList.add('fx-off');
    return; // touch / reduced / coarse → graceful no-op
  }

  var doc = document, root = doc.documentElement, body = doc.body;
  root.classList.add('fx-on');

  var DPR = Math.min(window.devicePixelRatio || 1, 1.5);
  var VW = window.innerWidth, VH = window.innerHeight;

  /* ── Shared rAF ticker (runs only while tasks exist) ────────────────── */
  var tasks = new Set(), rafId = null, lastT = 0;
  function tick(t) {
    rafId = null;
    var dt = Math.min((t - lastT) / 16.7, 3) || 1;
    lastT = t;
    tasks.forEach(function (fn) {
      try {
        if (fn(t, dt) === false) tasks.delete(fn);
      } catch (e) {
        tasks.delete(fn);
      }
    });
    if (tasks.size && !doc.hidden) rafId = requestAnimationFrame(tick);
  }

  function addTask(fn) {
    tasks.add(fn);
    if (!rafId && !doc.hidden) {
      lastT = performance.now();
      rafId = requestAnimationFrame(tick);
    }
  }

  function dropTask(fn) {
    tasks.delete(fn);
  }

  doc.addEventListener('visibilitychange', function () {
    if (doc.hidden) {
      if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    } else if (tasks.size) {
      addTask(function () { return tasks.size > 0; });
    }
  });

  function lerp(a, b, k, dt) { return a + (b - a) * Math.min(k * dt, 1); }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  function el(tag, cls, id) {
    var n = doc.createElement(tag);
    if (cls) n.className = cls;
    if (id) n.id = id;
    n.setAttribute('aria-hidden', 'true');
    return n;
  }

  function ensure(id, tag, cls, parent) {
    var n = doc.getElementById(id);
    if (!n) {
      n = el(tag, cls, id);
      (parent || body).appendChild(n);
    }
    return n;
  }

  /* ── 1 · Standard system cursor preserved (custom dot/ring removed) ── */

  /* ── 2 · Magnetic CTAs (spring return) ────────────────────────────────── */
  (function () {
    var magnets = [];
    function attach(btn) {
      if (!btn || btn.__fxMag) return;
      btn.__fxMag = true;
      var m = { btn: btn, r: null, tx: 0, ty: 0, x: 0, y: 0 };
      magnets.push(m);
      btn.addEventListener('pointerenter', function () {
        if (!btn.disabled) m.r = btn.getBoundingClientRect();
      }, { passive: true });
      btn.addEventListener('pointermove', function (e) {
        if (!m.r || btn.disabled) return;
        m.tx = clamp((e.clientX - (m.r.left + m.r.width / 2)) * 0.14, -6, 6);
        m.ty = clamp((e.clientY - (m.r.top + m.r.height / 2)) * 0.18, -6, 6);
        addTask(frame);
      }, { passive: true });
      btn.addEventListener('pointerleave', function () {
        m.tx = 0; m.ty = 0; m.r = null;
        addTask(frame);
      }, { passive: true });
    }

    function frame(t, dt) {
      var busy = false;
      for (var i = 0; i < magnets.length; i++) {
        var m = magnets[i];
        m.x = lerp(m.x, m.tx, 0.22, dt);
        m.y = lerp(m.y, m.ty, 0.22, dt);
        if (Math.abs(m.x - m.tx) > 0.05 || Math.abs(m.y - m.ty) > 0.05) busy = true;
        else { m.x = m.tx; m.y = m.ty; }
        m.btn.style.setProperty('--mag-x', m.x.toFixed(2) + 'px');
        m.btn.style.setProperty('--mag-y', m.y.toFixed(2) + 'px');
      }
      return busy;
    }

    doc.querySelectorAll('.btn-run, .mic-btn, .quick-demo-btn').forEach(attach);
    new MutationObserver(function (muts) {
      muts.forEach(function (mu) {
        if (mu.addedNodes) {
          mu.addedNodes.forEach(function (n) {
            if (n.nodeType === 1) {
              if (n.matches && n.matches('.btn-run, .mic-btn, .quick-demo-btn')) attach(n);
              if (n.querySelectorAll) n.querySelectorAll('.btn-run, .mic-btn, .quick-demo-btn').forEach(attach);
            }
          });
        }
      });
    }).observe(body, { childList: true, subtree: true });
  })();

  /* ── 3 · Subtle 3D Card Tilt (perspective) ─────────────────────────── */
  (function () {
    var TILT_SEL = '.card-box.card-module, .arch-card, .tax-card';
    var MAX = 3.5;

    function attachTilt(el) {
      if (!el || el.__fxTilt) return;
      el.__fxTilt = true;
      el.addEventListener('pointermove', function (e) {
        var r = el.getBoundingClientRect();
        var px = (e.clientX - r.left) / r.width - 0.5;
        var py = (e.clientY - r.top) / r.height - 0.5;
        var tx = (py * -MAX).toFixed(2) + 'deg';
        var ty = (px * MAX).toFixed(2) + 'deg';
        el.style.setProperty('--fx-tilt-x', tx);
        el.style.setProperty('--fx-tilt-y', ty);
      }, { passive: true });
      el.addEventListener('pointerleave', function () {
        el.style.setProperty('--fx-tilt-x', '0deg');
        el.style.setProperty('--fx-tilt-y', '0deg');
      }, { passive: true });
    }

    doc.querySelectorAll(TILT_SEL).forEach(attachTilt);
    new MutationObserver(function (muts) {
      muts.forEach(function (mu) {
        if (mu.addedNodes) {
          mu.addedNodes.forEach(function (n) {
            if (n.nodeType === 1) {
              if (n.matches && n.matches(TILT_SEL)) attachTilt(n);
              if (n.querySelectorAll) n.querySelectorAll(TILT_SEL).forEach(attachTilt);
            }
          });
        }
      });
    }).observe(body, { childList: true, subtree: true });
  })();

  /* ── 4 · Scroll reveals — IO lazy, staggered, conflict-free ─────────── */
  (function () {
    var SEL = '.flow-step-card, .arch-card, .dt-stat, .telem-item, .analytics-card, .ckpt-card';
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          en.target.classList.add('fx-inview');
          io.unobserve(en.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });

    function register(n) {
      if (n.__fxReveal || !n.matches || !n.matches(SEL)) return;
      if (n.parentElement && n.parentElement.classList.contains('panel-inner')) return;
      n.__fxReveal = true;
      var sibs = n.parentElement ? Array.prototype.filter.call(n.parentElement.children, function (c) { return c.matches(SEL); }) : [n];
      var idx = Math.min(sibs.indexOf(n), 5);
      n.style.setProperty('--fx-d', (idx * 70) + 'ms');
      n.classList.add('fx-reveal');
      io.observe(n);
    }

    doc.querySelectorAll(SEL).forEach(register);
    var grid = doc.getElementById('checkpoint-grid');
    if (grid) new MutationObserver(function (muts) {
      muts.forEach(function (mu) { mu.addedNodes.forEach(function (n) { if (n.nodeType === 1) register(n); }); });
    }).observe(grid, { childList: true });
  })();

  /* ── 5 · Hero pointer parallax + glow · aurora scroll parallax ──────── */
  (function () {
    var showcase = doc.querySelector('.banner-visual-showcase');
    var frameEl  = doc.querySelector('.banner-img-frame');
    var glow     = doc.querySelector('.hero-glow') || ensure('fx-hero-glow', 'div', 'hero-glow', showcase);

    if (showcase && frameEl) {
      var active = false, rect = null;
      var tx = 0, ty = 0, x = 0, y = 0, gx = 210, gy = 110, cgx = gx, cgy = gy;
      new IntersectionObserver(function (en) { active = en[0].isIntersecting; }, { threshold: 0.05 }).observe(showcase);
      showcase.addEventListener('pointerenter', function () { rect = showcase.getBoundingClientRect(); }, { passive: true });
      showcase.addEventListener('pointermove', function (e) {
        if (!active || !rect) return;
        var px = e.clientX - rect.left, py = e.clientY - rect.top;
        tx = clamp((px / rect.width - 0.5) * 12, -6, 6);
        ty = clamp((py / rect.height - 0.5) * 8, -4, 4);
        gx = px; gy = py;
        addTask(frame);
      }, { passive: true });
      showcase.addEventListener('pointerleave', function () { tx = 0; ty = 0; addTask(frame); }, { passive: true });

      function frame(t, dt) {
        x = lerp(x, tx, 0.14, dt);
        y = lerp(y, ty, 0.14, dt);
        cgx = lerp(cgx, gx, 0.2, dt);
        cgy = lerp(cgy, gy, 0.2, dt);
        frameEl.style.transform = 'translate3d(' + x.toFixed(2) + 'px,' + y.toFixed(2) + 'px,0)';
        frameEl.style.setProperty('--fx-parallax-x', (x * 0.7).toFixed(2) + 'px');
        frameEl.style.setProperty('--fx-parallax-y', (y * 0.7).toFixed(2) + 'px');
        if (glow) {
          glow.style.setProperty('--gx', cgx.toFixed(1) + 'px');
          glow.style.setProperty('--gy', cgy.toFixed(1) + 'px');
        }
        return Math.abs(x - tx) > 0.03 || Math.abs(y - ty) > 0.03 ||
               Math.abs(cgx - gx) > 0.3 || Math.abs(cgy - gy) > 0.3;
      }
    }

    var auroras = doc.querySelectorAll('.aurora');
    var progress = ensure('fx-scroll-progress', 'div', 'fx-scroll-progress');
    if (!progress.firstElementChild) progress.appendChild(el('span'));
    var bar = progress.firstElementChild;
    var rates = [-0.035, -0.055, -0.02], scrollDirty = true, maxScroll = 1, frameCount = 0;

    function measure() {
      VW = window.innerWidth; VH = window.innerHeight;
      maxScroll = Math.max(root.scrollHeight - VH, 1);
    }
    window.addEventListener('scroll', function () { scrollDirty = true; addTask(onScroll); }, { passive: true });
    window.addEventListener('resize', function () { measure(); scrollDirty = true; addTask(onScroll); }, { passive: true });

    function onScroll() {
      if (!scrollDirty) return false;
      scrollDirty = false;
      var sy = window.scrollY || 0;
      if (++frameCount % 30 === 0) maxScroll = Math.max(root.scrollHeight - VH, 1);
      bar.style.transform = 'scaleX(' + clamp(sy / maxScroll, 0, 1).toFixed(4) + ')';
      var capped = Math.min(sy, 900);
      for (var i = 0; i < auroras.length; i++) {
        auroras[i].style.translate = '0 ' + (capped * (rates[i] || -0.03)).toFixed(1) + 'px';
      }
      return false;
    }
    measure();
    addTask(onScroll);
  })();

  /* ── 6 · Spectrogram Cursor Spotlight ───────────────────────────────── */
  (function () {
    function initSpotlight(parent) {
      (parent || doc).querySelectorAll('.spec-canvas-wrap').forEach(function (wrap) {
        if (wrap.__fxSpot) return;
        wrap.__fxSpot = true;
        wrap.addEventListener('pointermove', function (e) {
          var r = wrap.getBoundingClientRect();
          var x = ((e.clientX - r.left) / r.width * 100).toFixed(2);
          var y = ((e.clientY - r.top) / r.height * 100).toFixed(2);
          wrap.style.setProperty('--fx-cursor-x', x + '%');
          wrap.style.setProperty('--fx-cursor-y', y + '%');
        }, { passive: true });
      });
    }
    initSpotlight(doc);
    var panelStudio = doc.getElementById('view-studio');
    if (panelStudio) new MutationObserver(function () { initSpotlight(panelStudio); }).observe(panelStudio, { childList: true, subtree: true });
  })();

  /* ── 7 · Audio-reactive ambient canvas ───────────────────────────────── */
  var ambient = (function () {
    var cv = ensure('fx-ambient', 'canvas', 'fx-ambient');
    var ctx = cv.getContext('2d');
    var micBtn = doc.getElementById('mic-btn');
    var micFill = doc.getElementById('mic-level-fill');
    var N = 36, P = [], energy = 0, smooth = 0, idleFrames = 0, running = false, lastVarRead = 0, appEnergy = -1;

    function size() {
      VW = window.innerWidth; VH = window.innerHeight;
      cv.width = VW * DPR; cv.height = VH * DPR;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    }
    function seed() {
      P.length = 0;
      for (var i = 0; i < N; i++) P.push({
        x: Math.random() * VW, y: Math.random() * VH,
        r: 0.8 + Math.random() * 1.8,
        vx: (Math.random() - 0.5) * 0.12, vy: -0.06 - Math.random() * 0.22,
        tw: Math.random() * Math.PI * 2,
        hue: Math.random() < 0.82 ? '34,211,238' : '129,140,248'
      });
    }
    function readEnergy(t) {
      var e = 0;
      if (micBtn && micBtn.classList.contains('recording')) {
        var w = micFill && micFill.style.width ? parseFloat(micFill.style.width) : 0;
        e = 0.3 + (isFinite(w) ? w / 100 : 0) * 0.7;
      } else if (body.classList.contains('audio-playing')) {
        e = 0.3 + Math.sin(t / 480) * 0.1;
      }
      if (t - lastVarRead > 300) {
        lastVarRead = t;
        var v = parseFloat(getComputedStyle(root).getPropertyValue('--ambient-energy'));
        appEnergy = isFinite(v) ? v : -1;
      }
      if (appEnergy >= 0) e = Math.max(e, appEnergy);
      return clamp(e, 0, 1);
    }
    function frame(t, dt) {
      energy = readEnergy(t);
      smooth = lerp(smooth, energy, 0.08, dt);
      if (smooth < 0.02) {
        if (++idleFrames > 50) { ctx.clearRect(0, 0, VW, VH); running = false; return false; }
      } else idleFrames = 0;
      ctx.clearRect(0, 0, VW, VH);
      if (smooth >= 0.02) {
        for (var i = 0; i < N; i++) {
          var p = P[i];
          p.tw += 0.02 * dt;
          p.x += p.vx * dt * (1 + smooth * 2.4);
          p.y += p.vy * dt * (1 + smooth * 2.4);
          if (p.y < -8) { p.y = VH + 8; p.x = Math.random() * VW; }
          if (p.x < -8) p.x = VW + 8; else if (p.x > VW + 8) p.x = -8;
          var a = (0.10 + 0.3 * smooth) * (0.6 + 0.4 * Math.sin(p.tw));
          var r = p.r * (1 + smooth * 1.3);
          ctx.beginPath();
          ctx.arc(p.x, p.y, r, 0, 6.2832);
          ctx.fillStyle = 'rgba(' + p.hue + ',' + a.toFixed(3) + ')';
          ctx.fill();
        }
      }
      return true;
    }
    function wake() { if (!running) { running = true; idleFrames = 0; addTask(frame); } }

    size(); seed();
    window.addEventListener('resize', function () { size(); seed(); }, { passive: true });
    new MutationObserver(wake).observe(body, { attributes: true, attributeFilter: ['class'] });
    if (micBtn) new MutationObserver(wake).observe(micBtn, { attributes: true, attributeFilter: ['class'] });
    setInterval(function () {
      if (!running && (body.classList.contains('audio-playing') || (micBtn && micBtn.classList.contains('recording')))) wake();
    }, 1200);
    wake();
    return { wake: wake };
  })();

  /* ── 8 · Peak moment: result burst + guarded confidence count-up ────── */
  (function () {
    var out = doc.getElementById('result-output');
    if (!out) return;
    var cv = ensure('fx-burst', 'canvas', 'fx-burst');
    var ctx = cv.getContext('2d');
    var sparks = [], burstOn = false;

    function size() { cv.width = VW * DPR; cv.height = VH * DPR; ctx.setTransform(DPR, 0, 0, DPR, 0, 0); }
    size(); window.addEventListener('resize', size, { passive: true });

    function burst() {
      var card = out.querySelector('.primary-prediction-card');
      if (!card) return;
      var r = card.getBoundingClientRect();
      if (r.bottom < 0 || r.top > VH) return;
      var cx = r.left + Math.min(r.width * 0.5, 260), cy = r.top + r.height / 2;
      var colors = ['251,191,36', '251,191,36', '52,211,153', '34,211,238'];
      for (var i = 0; i < 28; i++) {
        var a = Math.random() * Math.PI * 2, sp = 1.4 + Math.random() * 3.4;
        sparks.push({
          x: cx, y: cy,
          vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 1.1,
          life: 1, decay: 0.016 + Math.random() * 0.02,
          r: 1.2 + Math.random() * 2.2,
          c: colors[(Math.random() * colors.length) | 0]
        });
      }
      if (!burstOn) { burstOn = true; addTask(frame); }
    }

    function frame(t, dt) {
      ctx.clearRect(0, 0, VW, VH);
      for (var i = sparks.length - 1; i >= 0; i--) {
        var s = sparks[i];
        s.x += s.vx * dt; s.y += s.vy * dt; s.vy += 0.045 * dt;
        s.life -= s.decay * dt;
        if (s.life <= 0) { sparks.splice(i, 1); continue; }
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r * s.life, 0, 6.2832);
        ctx.fillStyle = 'rgba(' + s.c + ',' + (s.life * 0.85).toFixed(3) + ')';
        ctx.fill();
      }
      if (!sparks.length) { ctx.clearRect(0, 0, VW, VH); burstOn = false; return false; }
      return true;
    }

    function countUp() {
      var node = doc.getElementById('conf-pct');
      if (!node) return;
      var original = (node.textContent || '').trim();
      var m = original.match(/^(\d+(?:\.\d+)?)\s*%$/);
      if (!m) return;
      var target = parseFloat(m[1]);
      if (!(target > 0)) return;
      var decimals = (m[1].split('.')[1] || '').length;
      var mine = null, aborted = false;
      var mo = new MutationObserver(function () {
        if (mine !== null && (node.textContent || '').trim() !== mine) { aborted = true; }
      });
      mo.observe(node, { childList: true, characterData: true, subtree: true });
      var t0 = performance.now(), DUR = 720;
      addTask(function (t) {
        if (aborted) { mo.disconnect(); return false; }
        var k = clamp((t - t0) / DUR, 0, 1);
        var eased = 1 - Math.pow(1 - k, 3);
        mine = (target * eased).toFixed(decimals) + '%';
        node.textContent = mine;
        if (k >= 1) {
          mine = original; node.textContent = original;
          setTimeout(function () { mo.disconnect(); }, 0);
          return false;
        }
        return true;
      });
    }

    var wasHidden = out.hasAttribute('hidden');
    new MutationObserver(function () {
      var hidden = out.hasAttribute('hidden');
      if (wasHidden && !hidden) {
        ambient.wake();
        setTimeout(function () { burst(); countUp(); }, 150);
      }
      wasHidden = hidden;
    }).observe(out, { attributes: true, attributeFilter: ['hidden'] });
  })();

  /* ── 9 · View Switcher & Runtime preference safety ─────────────────── */
  function teardown() {
    tasks.clear();
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    root.classList.remove('fx-on');
    doc.querySelectorAll('.fx-reveal').forEach(function (n) { n.classList.add('fx-inview'); });
  }

  if (reducedMQ.addEventListener) reducedMQ.addEventListener('change', function (e) { if (e.matches) teardown(); });
  window.addEventListener('pagehide', teardown);
})();
