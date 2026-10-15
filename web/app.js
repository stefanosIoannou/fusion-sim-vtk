/* FusionScope — vtk.js viewer for EFIT equilibria from DIII-D and MAST.
 *
 * Coordinates: physical (R, φ, Z) → world (R cosφ, R sinφ, Z), Z up.
 * Per frame, ψN (0 at the magnetic axis, 1 at the LCFS) is contoured with vtkImageMarchingSquares,
 * and each contour is revolved toroidally into a flux surface.
 */
(() => {
  'use strict';

  const vtkGenericRenderWindow = vtk.Rendering.Misc.vtkGenericRenderWindow;
  const vtkPolyData = vtk.Common.DataModel.vtkPolyData;
  const vtkImageData = vtk.Common.DataModel.vtkImageData;
  const vtkDataArray = vtk.Common.Core.vtkDataArray;
  const vtkActor = vtk.Rendering.Core.vtkActor;
  const vtkMapper = vtk.Rendering.Core.vtkMapper;
  const vtkColorTransferFunction = vtk.Rendering.Core.vtkColorTransferFunction;
  const vtkImageMarchingSquares = vtk.Filters.General.vtkImageMarchingSquares;
  const vtkWarpScalar = vtk.Filters.General.vtkWarpScalar;

  const OUTSIDE = 1.6;          // ψN assigned outside the LCFS so contours < 1 stay inside the plasma
  const TERRAIN_CAP = 2.5;      // terrain height is (cap − ψN), so the magnetic axis is the summit
  const MAX_LEVELS = 14;

  const $ = (id) => document.getElementById(id);
  const shots = {};
  const loaders = {};
  window.FusionScope = {
    register(d) { shots[d.id] = d; if (loaders[d.id]) loaders[d.id](d); },
  };

  // ---------------------------------------------------------------- colours
  const PSI_STOPS = [[0, '#fcffa4'], [0.25, '#fca50a'], [0.5, '#dd513a'], [0.75, '#932667'], [1, '#420a68'], [1.6, '#12061f']];
  const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  function psiColor(v) {
    for (let i = 1; i < PSI_STOPS.length; i++) {
      const [x1, c1] = PSI_STOPS[i];
      if (v <= x1) {
        const [x0, c0] = PSI_STOPS[i - 1];
        const t = Math.max(0, (v - x0) / (x1 - x0));
        const a = hex(c0), b = hex(c1);
        return a.map((ai, k) => ai + (b[k] - ai) * t);
      }
    }
    return hex(PSI_STOPS[PSI_STOPS.length - 1][1]);
  }
  function makePsiLUT() {
    const lut = vtkColorTransferFunction.newInstance();
    PSI_STOPS.forEach(([x, c]) => lut.addRGBPoint(x, ...hex(c)));
    lut.addRGBPoint(TERRAIN_CAP, ...hex('#07030d'));
    return lut;
  }
  const COIL_NEG = '#3b7bff', COIL_ZERO = '#7c828c', COIL_POS = '#ff5a36';
  function coilColorCss(v, imax) {
    const t = Math.max(-1, Math.min(1, v / imax));
    const a = hex(COIL_ZERO), b = hex(t < 0 ? COIL_NEG : COIL_POS);
    const c = a.map((ai, k) => Math.round(255 * (ai + (b[k] - ai) * Math.abs(t))));
    return `rgb(${c.join(',')})`;
  }

  // ---------------------------------------------------------------- geometry helpers
  const arr = (name, values, nc = 1) => vtkDataArray.newInstance({ name, values, numberOfComponents: nc });

  function setPoly(pd, points, polys, normals, scalars) {
    pd.getPoints().setData(points, 3);
    pd.getPolys().setData(polys);
    if (normals) pd.getPointData().setNormals(arr('Normals', normals, 3));
    if (scalars) pd.getPointData().setScalars(arr('Scalars', scalars));
    pd.modified();
  }

  /* Revolve 2-D segments (rz = [r0,z0,r1,z1,...], segs = [a,b,...]) about the Z axis over [phi0, phi1].
   * centers = [r,z] per segment, used to orient normals outward so shading is smooth. */
  function revolve(rz, segs, centers, phi0, phi1, nphi) {
    const np = rz.length / 2, ring = nphi + 1, nseg = segs.length / 2;
    const n2 = new Float32Array(np * 2);
    for (let s = 0; s < nseg; s++) {
      const a = segs[2 * s], b = segs[2 * s + 1];
      let nr = rz[2 * b + 1] - rz[2 * a + 1], nz = -(rz[2 * b] - rz[2 * a]);
      const mr = 0.5 * (rz[2 * a] + rz[2 * b]) - centers[2 * s];
      const mz = 0.5 * (rz[2 * a + 1] + rz[2 * b + 1]) - centers[2 * s + 1];
      if (nr * mr + nz * mz < 0) { nr = -nr; nz = -nz; }
      const L = Math.hypot(nr, nz) || 1;
      n2[2 * a] += nr / L; n2[2 * a + 1] += nz / L;
      n2[2 * b] += nr / L; n2[2 * b + 1] += nz / L;
    }
    const cos = new Float32Array(ring), sin = new Float32Array(ring);
    for (let k = 0; k < ring; k++) {
      const p = phi0 + (phi1 - phi0) * k / nphi;
      cos[k] = Math.cos(p); sin[k] = Math.sin(p);
    }
    const pts = new Float32Array(np * ring * 3), nrm = new Float32Array(np * ring * 3);
    for (let i = 0; i < np; i++) {
      const r = rz[2 * i], z = rz[2 * i + 1];
      const L = Math.hypot(n2[2 * i], n2[2 * i + 1]) || 1, nr = n2[2 * i] / L, nz = n2[2 * i + 1] / L;
      for (let k = 0; k < ring; k++) {
        const o = (i * ring + k) * 3;
        pts[o] = r * cos[k]; pts[o + 1] = r * sin[k]; pts[o + 2] = z;
        nrm[o] = nr * cos[k]; nrm[o + 1] = nr * sin[k]; nrm[o + 2] = nz;
      }
    }
    const polys = new Uint32Array(nseg * nphi * 5);
    let o = 0;
    for (let s = 0; s < nseg; s++) {
      const a = segs[2 * s] * ring, b = segs[2 * s + 1] * ring;
      for (let k = 0; k < nphi; k++) {
        polys[o++] = 4; polys[o++] = a + k; polys[o++] = b + k; polys[o++] = b + k + 1; polys[o++] = a + k + 1;
      }
    }
    return { pts, nrm, polys, ring };
  }

  /* A closed or open (r,z) curve placed in the poloidal plane at angle phi, as a polyline. */
  function curveAt(rz, phi, closed) {
    const n = rz.length / 2, c = Math.cos(phi), s = Math.sin(phi);
    const pts = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { pts[3 * i] = rz[2 * i] * c; pts[3 * i + 1] = rz[2 * i] * s; pts[3 * i + 2] = rz[2 * i + 1]; }
    const ids = [n + (closed ? 1 : 0)];
    for (let i = 0; i < n; i++) ids.push(i);
    if (closed) ids.push(0);
    return { pts, ids };
  }

  function pointInPolygon(x, y, pr, pz) {
    let inside = false;
    for (let i = 0, j = pr.length - 1; i < pr.length; j = i++) {
      if ((pz[i] > y) !== (pz[j] > y) && x < (pr[j] - pr[i]) * (y - pz[i]) / (pz[j] - pz[i]) + pr[i]) inside = !inside;
    }
    return inside;
  }

  function linesToSegments(pd) {
    const xyz = pd.getPoints().getData(), lines = pd.getLines().getData();
    const rz = new Float32Array((xyz.length / 3) * 2);
    for (let i = 0; i < xyz.length / 3; i++) { rz[2 * i] = xyz[3 * i]; rz[2 * i + 1] = xyz[3 * i + 1]; }
    const segs = [];
    for (let i = 0; i < lines.length;) {
      const n = lines[i];
      for (let k = 1; k < n; k++) segs.push(lines[i + k], lines[i + k + 1]);
      i += n + 1;
    }
    return { rz, segs };
  }

  // ---------------------------------------------------------------- 3D tokamak view
  const main = vtkGenericRenderWindow.newInstance({ background: hex('#0a0d14') });
  main.setContainer($('view3d'));
  const ren = main.getRenderer(), rw = main.getRenderWindow();

  function addActor(renderer, pd, opts = {}) {
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(pd);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    const p = actor.getProperty();
    p.setAmbient(opts.ambient ?? 0.18); p.setDiffuse(opts.diffuse ?? 0.85);
    p.setSpecular(opts.specular ?? 0.25); p.setSpecularPower(opts.specularPower ?? 30);
    if (opts.color) { p.setColor(...opts.color); mapper.setScalarVisibility(false); }
    if (opts.lut) {
      mapper.setLookupTable(opts.lut); mapper.setUseLookupTableScalarRange(true);
      mapper.setScalarModeToUsePointData(); mapper.setColorModeToMapScalars();
    }
    if (opts.opacity !== undefined) p.setOpacity(opts.opacity);
    if (opts.lineWidth) p.setLineWidth(opts.lineWidth);
    renderer.addActor(actor);
    return { actor, mapper, pd };
  }

  const psiLUT = makePsiLUT();
  const coilLUT = vtkColorTransferFunction.newInstance();

  const levelActors = [];
  for (let i = 0; i < MAX_LEVELS; i++) levelActors.push(addActor(ren, vtkPolyData.newInstance(), { color: [1, 1, 1] }));
  const lcfsShell = addActor(ren, vtkPolyData.newInstance(), { color: hex('#c9a6ff'), opacity: 0.16, specular: 0.6 });
  const caps = addActor(ren, vtkPolyData.newInstance(), { lut: psiLUT, ambient: 0.45, diffuse: 0.6, specular: 0 });
  const edgeLines = addActor(ren, vtkPolyData.newInstance(), { color: hex('#7fd8ff'), lineWidth: 2, ambient: 1, diffuse: 0 });
  const coils = addActor(ren, vtkPolyData.newInstance(), { lut: coilLUT, ambient: 0.45, diffuse: 0.7, specular: 0.4 });

  const msImage = vtkImageData.newInstance();
  const msScalars = arr('psiN', new Float32Array(1));
  const marchers = [];
  for (let i = 0; i < MAX_LEVELS; i++) {
    const ms = vtkImageMarchingSquares.newInstance({ contourValues: [0.5], slice: 0 });
    if (ms.setMergePoints) ms.setMergePoints(true);
    ms.setInputData(msImage);
    marchers.push(ms);
  }

  // ---------------------------------------------------------------- terrain view
  const terr = vtkGenericRenderWindow.newInstance({ background: hex('#0a0d14') });
  terr.setContainer($('terrain'));
  const tren = terr.getRenderer(), trw = terr.getRenderWindow();
  const terrainPD = vtkPolyData.newInstance();
  const warp = vtkWarpScalar.newInstance({ useNormal: true, normal: [0, 0, 1] });
  warp.setInputData(terrainPD);
  const terrainActor = vtkActor.newInstance();
  const terrainMapper = vtkMapper.newInstance({ interpolateScalarsBeforeMapping: true });
  terrainMapper.setInputConnection(warp.getOutputPort());
  terrainMapper.setLookupTable(psiLUT); terrainMapper.setUseLookupTableScalarRange(true);
  terrainActor.setMapper(terrainMapper);
  terrainActor.getProperty().set({ ambient: 0.25, diffuse: 0.8, specular: 0.15, specularPower: 20 });
  tren.addActor(terrainActor);
  const terrainLines = addActor(tren, vtkPolyData.newInstance(), { color: [0.95, 0.95, 1], opacity: 0.45, ambient: 1, diffuse: 0 });
  const terrainLCFS = addActor(tren, vtkPolyData.newInstance(), { color: hex('#7fd8ff'), lineWidth: 3, ambient: 1, diffuse: 0 });
  const terrainMS = vtkImageMarchingSquares.newInstance({ contourValues: [], slice: 0 });
  const terrainImage = vtkImageData.newInstance();
  const terrainImgScalars = arr('psiN', new Float32Array(1));
  terrainMS.setInputData(terrainImage);

  // ---------------------------------------------------------------- state
  const state = {
    shot: null, frame: 0, playing: false, selected: 0,
    show: { surfaces: true, lcfs: true, caps: false, coils: true },
    cut: 100, levels: 8, opacity: 1,
  };

  function prepare(d) {
    if (d._ready) return d;
    const bin = atob(d.psin.b64), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const u16 = new Uint16Array(u8.buffer), psin = new Float32Array(u16.length);
    const sc = (d.psin.hi - d.psin.lo) / 65535;
    for (let i = 0; i < u16.length; i++) psin[i] = d.psin.lo + u16[i] * sc;
    d.psinField = psin;
    d.nR = d.R.length; d.nZ = d.Z.length;
    d.dR = (d.R[d.nR - 1] - d.R[0]) / (d.nR - 1); d.dZ = (d.Z[d.nZ - 1] - d.Z[0]) / (d.nZ - 1);
    d.imax = 1;
    d.circuits.forEach((c) => c.atEfit.forEach((v) => { d.imax = Math.max(d.imax, Math.abs(v)); }));
    d.shape = d.lcfs.map(({ r, z }) => {
      if (!r.length) return null;
      let iMaxZ = 0, iMinZ = 0, rmin = Infinity, rmax = -Infinity;
      r.forEach((ri, i) => {
        rmin = Math.min(rmin, ri); rmax = Math.max(rmax, ri);
        if (z[i] > z[iMaxZ]) iMaxZ = i;
        if (z[i] < z[iMinZ]) iMinZ = i;
      });
      const a = (rmax - rmin) / 2, r0 = (rmax + rmin) / 2;
      return { kappa: (z[iMaxZ] - z[iMinZ]) / (2 * a), deltaU: (r0 - r[iMaxZ]) / a, deltaL: (r0 - r[iMinZ]) / a, a };
    });
    // pick the most active circuit as the default plotted coil
    let best = 0, bestVal = -1;
    d.circuits.forEach((c, i) => { const m = Math.max(...c.trace.map(Math.abs)); if (m > bestVal) { bestVal = m; best = i; } });
    d.defaultCircuit = best;
    d._ready = true;
    return d;
  }

  // ---------------------------------------------------------------- per-shot geometry
  let coilPointCircuit = null, coilScalars = null;

  function buildCoils(d) {
    const nphi = d.coils.length > 100 ? 40 : 72;
    const phi0 = state.cut * Math.PI / 180, phi1 = 2 * Math.PI;
    const rz = [], segs = [], centers = [], segCircuit = [];
    d.coils.forEach(({ c, p }) => {
      const cr = (p[0] + p[2] + p[4] + p[6]) / 4, cz = (p[1] + p[3] + p[5] + p[7]) / 4;
      for (let e = 0; e < 4; e++) {
        const f = (e + 1) % 4, base = rz.length / 2;
        rz.push(p[2 * e], p[2 * e + 1], p[2 * f], p[2 * f + 1]);   // duplicate corners → crisp edges
        segs.push(base, base + 1); centers.push(cr, cz); segCircuit.push(c);
      }
    });
    const g = revolve(Float32Array.from(rz), segs, centers, phi0, phi1, nphi);
    coilPointCircuit = new Uint16Array(rz.length / 2 * g.ring);
    for (let s = 0; s < segs.length / 2; s++) {
      for (const pi of [segs[2 * s], segs[2 * s + 1]]) coilPointCircuit.fill(segCircuit[s], pi * g.ring, (pi + 1) * g.ring);
    }
    coilScalars = new Float32Array(coilPointCircuit.length);
    setPoly(coils.pd, g.pts, g.polys, g.nrm, coilScalars);

    coilLUT.removeAllPoints();
    coilLUT.addRGBPoint(-d.imax, ...hex(COIL_NEG));
    coilLUT.addRGBPoint(0, ...hex(COIL_ZERO));
    coilLUT.addRGBPoint(d.imax, ...hex(COIL_POS));
    $('leg-imin').textContent = `−${Math.round(d.imax)} kA`;
    $('leg-imax').textContent = `+${Math.round(d.imax)} kA`;
  }

  function buildTerrainGrid(d) {
    const { nR, nZ, R, Z } = d;
    const pts = new Float32Array(nR * nZ * 3);
    for (let j = 0; j < nZ; j++) for (let i = 0; i < nR; i++) {
      const o = (j * nR + i) * 3; pts[o] = R[i]; pts[o + 1] = Z[j]; pts[o + 2] = 0;
    }
    const polys = new Uint32Array((nR - 1) * (nZ - 1) * 5);
    let o = 0;
    for (let j = 0; j < nZ - 1; j++) for (let i = 0; i < nR - 1; i++) {
      const a = j * nR + i;
      polys[o++] = 4; polys[o++] = a; polys[o++] = a + 1; polys[o++] = a + 1 + nR; polys[o++] = a + nR;
    }
    terrainPD.getPoints().setData(pts, 3);
    terrainPD.getPolys().setData(polys);
    terrainPD.getPointData().setScalars(arr('height', new Float32Array(nR * nZ)));
    terrainPD.getPointData().addArray(arr('psiN', new Float32Array(nR * nZ)));
    terrainMapper.setScalarModeToUsePointFieldData();
    terrainMapper.setColorByArrayName('psiN');
    terrainMapper.setColorModeToMapScalars();
    const zspan = Z[nZ - 1] - Z[0];
    d.terrainScale = 0.22 * zspan / TERRAIN_CAP;
    warp.setScaleFactor(d.terrainScale);
    for (const img of [msImage, terrainImage]) {
      img.setDimensions(nR, nZ, 1); img.setOrigin(R[0], Z[0], 0); img.setSpacing(d.dR, d.dZ, 1);
    }
    msScalars.setData(new Float32Array(nR * nZ), 1);
    msImage.getPointData().setScalars(msScalars);
    terrainImgScalars.setData(new Float32Array(nR * nZ), 1);
    terrainImage.getPointData().setScalars(terrainImgScalars);
    const levels = [];
    for (let v = 0.1; v < TERRAIN_CAP; v += 0.1) levels.push(+v.toFixed(2));
    terrainMS.setContourValues(levels);
  }

  // ---------------------------------------------------------------- per-frame update
  function updateFrame() {
    const d = state.shot, k = state.frame;
    if (!d) return;
    const { nR, nZ, R, Z } = d, n = nR * nZ;
    const field = d.psinField.subarray(k * n, (k + 1) * n);
    const lc = d.lcfs[k];
    const phi0 = state.cut * Math.PI / 180, phi1 = 2 * Math.PI;
    const nphi = Math.max(8, Math.round((phi1 - phi0) / (2 * Math.PI) * 96));

    // mask ψN outside the LCFS (private-flux region below the X-point also has ψN < 1)
    const masked = msScalars.getData(), inside = new Uint8Array(n);
    for (let j = 0; j < nZ; j++) for (let i = 0; i < nR; i++) {
      const p = j * nR + i;
      inside[p] = lc.r.length > 2 && pointInPolygon(R[i], Z[j], lc.r, lc.z) ? 1 : 0;
      masked[p] = inside[p] ? Math.min(field[p], 0.999) : Math.max(field[p], OUTSIDE);
    }
    msScalars.modified(); msImage.modified();

    // nested flux surfaces
    const ax = [d.axis.r[k], d.axis.z[k]];
    for (let li = 0; li < MAX_LEVELS; li++) {
      const la = levelActors[li];
      const on = state.show.surfaces && li < state.levels;
      la.actor.setVisibility(on);
      if (!on) continue;
      const level = (li + 1) / (state.levels + 1);
      marchers[li].setContourValues([level]);
      const { rz, segs } = linesToSegments(marchers[li].getOutputData());
      if (!segs.length) { la.actor.setVisibility(false); continue; }
      const centers = new Float32Array(segs.length);
      for (let s = 0; s < segs.length / 2; s++) { centers[2 * s] = ax[0]; centers[2 * s + 1] = ax[1]; }
      const g = revolve(rz, segs, centers, phi0, phi1, nphi);
      setPoly(la.pd, g.pts, g.polys, g.nrm);
      la.actor.getProperty().setColor(...psiColor(level));
      la.actor.getProperty().setOpacity(state.opacity);
    }

    // LCFS shell + its outline on both cut planes + magnetic axis circle
    const lrz = new Float32Array(lc.r.length * 2);
    lc.r.forEach((r, i) => { lrz[2 * i] = r; lrz[2 * i + 1] = lc.z[i]; });
    const nl = lc.r.length;
    if (nl > 2) {
      const segs = [], centers = [];
      for (let i = 0; i < nl; i++) { segs.push(i, (i + 1) % nl); centers.push(ax[0], ax[1]); }
      const g = revolve(lrz, segs, centers, phi0, phi1, nphi);
      setPoly(lcfsShell.pd, g.pts, g.polys, g.nrm);
    }
    lcfsShell.actor.setVisibility(state.show.lcfs && nl > 2);

    const linePts = [], lineIds = [];
    const pushCurve = (c) => {
      const off = linePts.length / 3;
      linePts.push(...c.pts);
      lineIds.push(c.ids[0], ...c.ids.slice(1).map((x) => x + off));
    };
    if (nl > 2 && state.cut > 0) { pushCurve(curveAt(lrz, phi0, true)); pushCurve(curveAt(lrz, phi1, true)); }
    const axisRing = [];
    for (let kk = 0; kk <= nphi; kk++) axisRing.push(ax[0], ax[1]);
    if (state.show.surfaces) {
      const off = linePts.length / 3;
      for (let kk = 0; kk <= nphi; kk++) {
        const p = phi0 + (phi1 - phi0) * kk / nphi;
        linePts.push(ax[0] * Math.cos(p), ax[0] * Math.sin(p), ax[1]);
      }
      lineIds.push(nphi + 1, ...Array.from({ length: nphi + 1 }, (_, i) => i + off));
    }
    edgeLines.pd.getPoints().setData(Float32Array.from(linePts), 3);
    edgeLines.pd.getLines().setData(Uint32Array.from(lineIds));
    edgeLines.pd.modified();
    edgeLines.actor.setVisibility(state.show.lcfs || state.show.surfaces);

    // ψN cross-section on the two cut faces (cells fully inside the LCFS)
    caps.actor.setVisibility(state.show.caps && state.cut > 0);
    if (state.show.caps && state.cut > 0) {
      const planes = [phi0, phi1];
      const pts = new Float32Array(2 * n * 3), nrm = new Float32Array(2 * n * 3), sc = new Float32Array(2 * n);
      const polys = [];
      planes.forEach((phi, pl) => {
        const c = Math.cos(phi), s = Math.sin(phi), sign = pl === 0 ? 1 : -1;
        for (let j = 0; j < nZ; j++) for (let i = 0; i < nR; i++) {
          const p = j * nR + i, o = (pl * n + p) * 3;
          pts[o] = R[i] * c; pts[o + 1] = R[i] * s; pts[o + 2] = Z[j];
          nrm[o] = -s * sign; nrm[o + 1] = c * sign; nrm[o + 2] = 0;
          sc[pl * n + p] = Math.max(0, field[p]);
        }
        for (let j = 0; j < nZ - 1; j++) for (let i = 0; i < nR - 1; i++) {
          const a = j * nR + i;
          if (inside[a] && inside[a + 1] && inside[a + nR] && inside[a + nR + 1]) {
            const b = pl * n + a;
            polys.push(4, b, b + 1, b + 1 + nR, b + nR);
          }
        }
      });
      setPoly(caps.pd, pts, Uint32Array.from(polys), nrm, sc);
    }

    // coil currents → per-point scalars
    coils.actor.setVisibility(state.show.coils);
    for (let i = 0; i < coilScalars.length; i++) coilScalars[i] = d.circuits[coilPointCircuit[i]].atEfit[k];
    coils.pd.getPointData().getScalars().modified();
    coils.pd.modified();

    updateTerrain(field, lrz, nl);
    rw.render();
    updateHud();
    drawTraces();
    drawCoilBars();
  }

  function updateTerrain(field, lrz, nl) {
    const d = state.shot, n = field.length;
    const h = terrainPD.getPointData().getScalars().getData();
    const col = terrainPD.getPointData().getArrayByName('psiN').getData();
    const timg = terrainImgScalars.getData();
    for (let i = 0; i < n; i++) {
      const v = Math.min(Math.max(field[i], 0), TERRAIN_CAP);
      h[i] = TERRAIN_CAP - v; col[i] = v; timg[i] = v;
    }
    terrainPD.getPointData().getScalars().modified();
    terrainPD.getPointData().getArrayByName('psiN').modified();
    terrainPD.modified();
    terrainImgScalars.modified(); terrainImage.modified();

    // lift contour lines onto the terrain surface
    const out = terrainMS.getOutputData();
    const src = out.getPoints().getData(), lifted = new Float32Array(src.length);
    for (let i = 0; i < src.length; i += 3) {
      const fi = (src[i] - d.R[0]) / d.dR, fj = (src[i + 1] - d.Z[0]) / d.dZ;
      const v = sample(field, d.nR, d.nZ, fi, fj);
      lifted[i] = src[i]; lifted[i + 1] = src[i + 1]; lifted[i + 2] = (TERRAIN_CAP - Math.min(Math.max(v, 0), TERRAIN_CAP)) * d.terrainScale + 0.004;
    }
    terrainLines.pd.getPoints().setData(lifted, 3);
    terrainLines.pd.getLines().setData(out.getLines().getData());
    terrainLines.pd.modified();

    const lp = new Float32Array((nl + 1) * 3);
    for (let i = 0; i <= nl; i++) {
      const q = i % nl;
      lp[3 * i] = lrz[2 * q]; lp[3 * i + 1] = lrz[2 * q + 1]; lp[3 * i + 2] = (TERRAIN_CAP - 1) * d.terrainScale + 0.008;
    }
    terrainLCFS.pd.getPoints().setData(lp, 3);
    terrainLCFS.pd.getLines().setData(Uint32Array.from([nl + 1, ...Array.from({ length: nl + 1 }, (_, i) => i)]));
    terrainLCFS.pd.modified();
    trw.render();
  }

  function sample(f, nR, nZ, fi, fj) {
    const i = Math.min(Math.max(Math.floor(fi), 0), nR - 2), j = Math.min(Math.max(Math.floor(fj), 0), nZ - 2);
    const ti = fi - i, tj = fj - j, p = j * nR + i;
    return (1 - tj) * ((1 - ti) * f[p] + ti * f[p + 1]) + tj * ((1 - ti) * f[p + nR] + ti * f[p + nR + 1]);
  }

  // ---------------------------------------------------------------- cameras
  function resetCameras() {
    const d = state.shot;
    const mid = (state.cut / 2) * Math.PI / 180;
    const cam = ren.getActiveCamera();
    cam.setFocalPoint(0, 0, 0);
    const az = mid - 0.75;   // look into the cutaway at an angle so the nested shells show
    cam.setPosition(Math.cos(az) * 10, Math.sin(az) * 10, 6.5);
    cam.setViewUp(0, 0, 1);
    ren.resetCamera();
    cam.zoom(1.35);
    ren.resetCameraClippingRange();
    rw.render();

    const tc = tren.getActiveCamera();
    const rc = (d.R[0] + d.R[d.nR - 1]) / 2, zc = (d.Z[0] + d.Z[d.nZ - 1]) / 2;
    tc.setFocalPoint(rc, zc, 0);
    tc.setPosition(rc + 0.5, zc - 2.2, 7);
    tc.setViewUp(0, 1, 0);
    tren.resetCamera();
    tc.zoom(1.15);
    tren.resetCameraClippingRange();
    trw.render();
  }

  // ---------------------------------------------------------------- HUD + 2-D charts
  const fmt = (v, dp = 2) => (v === null || v === undefined || Number.isNaN(v) ? '–' : v.toFixed(dp));
  function updateHud() {
    const d = state.shot, k = state.frame, sh = d.shape[k];
    $('hud-machine').textContent = d.machine;
    $('hud-shot').textContent = `shot #${d.shot}`;
    $('hud-t').textContent = d.times[k].toFixed(0);
    const rows = [
      ['Plasma current Iₚ', `${fmt(d.ip.atEfit[k] / 1000, 3)} MA`],
      ['Toroidal field coil', `${fmt(d.tf.atEfit[k], 1)} kA`],
      ['q₉₅', fmt(d.scalars.q95[k])],
      ['β_N', fmt(d.scalars.betaN[k])],
      ['ℓᵢ', fmt(d.scalars.li[k])],
      ['Axis R, Z', `${fmt(d.axis.r[k], 3)}, ${fmt(d.axis.z[k], 3)} m`],
      ['Minor radius a', sh ? `${fmt(sh.a, 3)} m` : '–'],
      ['Elongation κ', sh ? fmt(sh.kappa) : '–'],
      ['Triangularity δ↑ / δ↓', sh ? `${fmt(sh.deltaU)} / ${fmt(sh.deltaL)}` : '–'],
    ];
    $('readout').innerHTML = rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join('');
    $('slider').value = k;
    $('frame-label').textContent = `${k + 1} / ${d.times.length}`;
  }

  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  const TRACE_PAD = { l: 44, r: 10, t: 4, b: 18 };
  function drawTraces() {
    const d = state.shot, cv = $('traces');
    const { ctx, w, h } = setupCanvas(cv);
    const c = d.circuits[state.selected];
    const series = [
      { label: 'Iₚ [MA]', t: d.traceTimes, y: d.ip.trace.map((v) => v / 1000), color: '#ffb347' },
      { label: 'q₉₅', t: d.times, y: d.scalars.q95, color: '#7fd8ff' },
      { label: 'β_N', t: d.times, y: d.scalars.betaN, color: '#b48cff' },
      { label: `${c.name} [kA]`, t: d.traceTimes, y: c.trace, color: coilColorCss(Math.max(...c.trace) > -Math.min(...c.trace) ? 1 : -1, 1) },
    ];
    const t0 = d.times[0], t1 = d.times[d.times.length - 1];
    const pw = w - TRACE_PAD.l - TRACE_PAD.r, rowH = (h - TRACE_PAD.t - TRACE_PAD.b) / series.length;
    const X = (t) => TRACE_PAD.l + (t - t0) / (t1 - t0) * pw;
    ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
    series.forEach((s, i) => {
      const top = TRACE_PAD.t + i * rowH + 3, bot = TRACE_PAD.t + (i + 1) * rowH - 3;
      let lo = Math.min(...s.y), hi = Math.max(...s.y);
      if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
      const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
      const Y = (v) => bot - (v - lo) / (hi - lo) * (bot - top);
      ctx.strokeStyle = '#263044'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(TRACE_PAD.l, bot + 2.5); ctx.lineTo(w - TRACE_PAD.r, bot + 2.5); ctx.stroke();
      if (lo < 0 && hi > 0) { ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(TRACE_PAD.l, Y(0)); ctx.lineTo(w - TRACE_PAD.r, Y(0)); ctx.stroke(); ctx.setLineDash([]); }
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
      s.t.forEach((t, j) => { const x = X(t), y = Y(s.y[j]); j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
      ctx.fillStyle = s.color; ctx.textAlign = 'right'; ctx.fillText(s.label, w - TRACE_PAD.r - 2, top + 9);
      ctx.fillStyle = '#8a93a6'; ctx.textAlign = 'right';
      ctx.fillText((hi - pad).toPrecision(3), TRACE_PAD.l - 4, top + 8);
      ctx.fillText((lo + pad).toPrecision(3), TRACE_PAD.l - 4, bot);
    });
    ctx.fillStyle = '#8a93a6'; ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) { const t = t0 + (t1 - t0) * i / 4; ctx.fillText(`${t.toFixed(0)}`, X(t), h - 5); }
    ctx.textAlign = 'right'; ctx.fillText('ms', TRACE_PAD.l - 6, h - 5);
    const xc = X(d.times[state.frame]);
    ctx.strokeStyle = '#ffffffcc'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(xc, TRACE_PAD.t); ctx.lineTo(xc, h - TRACE_PAD.b); ctx.stroke();
  }

  let barRows = [];
  function drawCoilBars() {
    const d = state.shot, k = state.frame, cv = $('coilbars');
    const { ctx, w, h } = setupCanvas(cv);
    const n = d.circuits.length, rowH = Math.min(22, (h - 8) / n), nameW = 70, valW = 54;
    const x0 = nameW + (w - nameW - valW) / 2, half = (w - nameW - valW) / 2 - 6;
    ctx.font = '11px system-ui, sans-serif';
    ctx.strokeStyle = '#263044'; ctx.beginPath(); ctx.moveTo(x0, 4); ctx.lineTo(x0, 4 + n * rowH); ctx.stroke();
    barRows = [];
    d.circuits.forEach((c, i) => {
      const v = c.atEfit[k], y = 4 + i * rowH, bw = Math.abs(v) / d.imax * half;
      if (i === state.selected) { ctx.fillStyle = '#ffffff12'; ctx.fillRect(0, y, w, rowH); }
      ctx.fillStyle = coilColorCss(v, d.imax);
      ctx.fillRect(v < 0 ? x0 - bw : x0, y + rowH * 0.2, bw, rowH * 0.6);
      ctx.fillStyle = i === state.selected ? '#ffffff' : '#c3c9d6'; ctx.textAlign = 'left';
      ctx.fillText(c.name, 8, y + rowH * 0.7);
      ctx.fillStyle = '#8a93a6'; ctx.textAlign = 'right';
      ctx.fillText(v.toFixed(1), w - 6, y + rowH * 0.7);
      barRows.push([y, y + rowH]);
    });
  }

  // ---------------------------------------------------------------- shot loading
  function loadShot(id) {
    if (shots[id]) return Promise.resolve(shots[id]);
    return new Promise((resolve, reject) => {
      loaders[id] = resolve;
      const s = document.createElement('script');
      s.src = `data/${id}.js`;
      s.onerror = () => reject(new Error(`could not load data/${id}.js — run build_data.py first`));
      document.head.appendChild(s);
    });
  }

  async function selectShot(id) {
    const status = $('status');
    status.hidden = false; status.textContent = `Loading ${id}…`;
    try {
      const d = prepare(await loadShot(id));
      state.shot = d; state.frame = Math.min(state.frame, d.times.length - 1);
      state.frame = Math.floor(d.times.length / 2);
      state.selected = d.defaultCircuit;
      document.querySelectorAll('#shots button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.id === id));
      $('slider').max = d.times.length - 1;
      buildTerrainGrid(d);
      buildCoils(d);
      updateFrame();
      resetCameras();
      status.hidden = true;
    } catch (e) {
      status.textContent = e.message; console.error(e);
    }
  }

  // ---------------------------------------------------------------- UI wiring
  function buildShotPicker() {
    const nav = $('shots');
    (window.FUSION_INDEX || []).forEach((s) => {
      const b = document.createElement('button');
      b.type = 'button'; b.dataset.id = s.id;
      b.innerHTML = `<span class="tag ${s.machine === 'MAST' ? 'mast' : 'd3d'}">${s.machine}</span>#${s.shot}`
        + `<small>${s.frames} frames · ${s.t0.toFixed(0)}–${s.t1.toFixed(0)} ms</small>`;
      b.addEventListener('click', () => selectShot(s.id));
      nav.appendChild(b);
    });
  }

  function seekFromCanvas(ev) {
    const d = state.shot; if (!d) return;
    const cv = $('traces'), rect = cv.getBoundingClientRect();
    const t0 = d.times[0], t1 = d.times[d.times.length - 1];
    const t = t0 + (ev.clientX - rect.left - TRACE_PAD.l) / (rect.width - TRACE_PAD.l - TRACE_PAD.r) * (t1 - t0);
    let best = 0;
    d.times.forEach((ti, i) => { if (Math.abs(ti - t) < Math.abs(d.times[best] - t)) best = i; });
    state.frame = best; updateFrame();
  }

  function wire() {
    const bind = (id, key, out, f = (x) => x) => {
      $(id).addEventListener('input', (e) => {
        state[key] = f(+e.target.value);
        if (out) $(out).textContent = key === 'cut' ? `${state[key]}°` : key === 'opacity' ? state[key].toFixed(2) : state[key];
        if (key === 'cut' && state.shot) buildCoils(state.shot);
        updateFrame();
      });
    };
    bind('opt-cut', 'cut', 'out-cut');
    bind('opt-levels', 'levels', 'out-levels');
    bind('opt-opacity', 'opacity', 'out-opacity');
    for (const key of ['surfaces', 'lcfs', 'caps', 'coils']) {
      $(`opt-${key}`).addEventListener('change', (e) => { state.show[key] = e.target.checked; updateFrame(); });
    }
    $('btn-camera').addEventListener('click', resetCameras);
    $('slider').addEventListener('input', (e) => { state.frame = +e.target.value; updateFrame(); });
    $('btn-play').addEventListener('click', togglePlay);

    let dragging = false;
    const tr = $('traces');
    tr.addEventListener('pointerdown', (e) => { dragging = true; tr.setPointerCapture(e.pointerId); seekFromCanvas(e); });
    tr.addEventListener('pointermove', (e) => { if (dragging) seekFromCanvas(e); });
    tr.addEventListener('pointerup', () => { dragging = false; });

    $('coilbars').addEventListener('click', (e) => {
      const y = e.clientY - $('coilbars').getBoundingClientRect().top;
      const i = barRows.findIndex(([a, b]) => y >= a && y < b);
      if (i >= 0) { state.selected = i; drawTraces(); drawCoilBars(); }
    });

    window.addEventListener('keydown', (e) => {
      if (!state.shot || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
      if (e.code === 'ArrowRight') { state.frame = Math.min(state.frame + 1, state.shot.times.length - 1); updateFrame(); }
      if (e.code === 'ArrowLeft') { state.frame = Math.max(state.frame - 1, 0); updateFrame(); }
    });

    const onResize = () => {
      main.resize(); terr.resize();
      if (state.shot) { drawTraces(); drawCoilBars(); rw.render(); trw.render(); }
    };
    window.addEventListener('resize', onResize);
    new ResizeObserver(onResize).observe($('terrain'));
  }

  let timer = null;
  function togglePlay() {
    state.playing = !state.playing;
    $('btn-play').textContent = state.playing ? '❚❚' : '▶';
    $('btn-play').setAttribute('aria-label', state.playing ? 'Pause' : 'Play');
    if (!state.playing) { clearTimeout(timer); return; }
    const step = () => {
      if (!state.playing || !state.shot) return;
      const t0 = performance.now();
      state.frame = (state.frame + 1) % state.shot.times.length;
      updateFrame();
      const wait = 1000 / +$('speed').value - (performance.now() - t0);
      timer = setTimeout(step, Math.max(0, wait));
    };
    step();
  }

  buildShotPicker();
  wire();
  main.resize(); terr.resize();
  const first = new URLSearchParams(location.search).get('shot') || (window.FUSION_INDEX && window.FUSION_INDEX[0].id);
  if (first) selectShot(first);
  else $('status').textContent = 'No shots found — run build_data.py';
})();
