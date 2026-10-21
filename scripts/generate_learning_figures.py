#!/usr/bin/env python3
"""Reproduce the learning article's theme-aware SVG figures using the stdlib.

The site builds from committed SVGs. Numerical checks independently compare the
analytic gradients with finite differences and verify all eight symmetries.
"""
from pathlib import Path
from html import escape
from itertools import product
from math import exp, log, log1p, tanh, sqrt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'content/figures'
INITIAL = [.5, 0., 0., .5, 0., 0., 1., 1., 0.]


def forward(theta, x=(1., 1.)):
    w11, w12, w21, w22, b1, b2, v1, v2, c = theta
    z = [tanh(w11*x[0]+w12*x[1]+b1), tanh(w21*x[0]+w22*x[1]+b2)]
    s = v1*z[0]+v2*z[1]+c
    return 1/(1+exp(-s)), z, s


def loss(theta, x=(1., 1.), y=1):
    s = forward(theta, x)[2]
    return max(s, 0)-y*s+log1p(exp(-abs(s)))


def gradient(theta, x=(1., 1.), y=1):
    p, z, _ = forward(theta, x)
    error = p-y
    hidden = [error*theta[6+j]*(1-z[j]**2) for j in range(2)]
    return [hidden[0]*x[0], hidden[0]*x[1], hidden[1]*x[0], hidden[1]*x[1],
            *hidden, error*z[0], error*z[1], error]


def updated():
    return [v-.1*g for v, g in zip(INITIAL, gradient(INITIAL))]


def symmetry(theta, swap=False, signs=(1, 1)):
    rows = [[theta[0], theta[1], theta[4], theta[6]],
            [theta[2], theta[3], theta[5], theta[7]]]
    if swap:
        rows.reverse()
    a, b = [[sign*v for v in row] for row, sign in zip(rows, signs)]
    return [a[0], a[1], b[0], b[1], a[2], b[2], a[3], b[3], theta[8]]


def verify_example():
    expected = (0.7159, 0.3342, 0.7443, 0.2954)
    actual = (forward(INITIAL)[0], loss(INITIAL), forward(updated())[0], loss(updated()))
    assert tuple(round(v, 4) for v in actual) == expected
    expected_parameters = [.5223, .0223, .0223, .5223, .0223, .0223, 1.0131, 1.0131, .0284]
    assert [round(v, 4) for v in updated()] == expected_parameters
    # Include unequal inputs and hidden states to avoid a symmetric test hiding indexing errors.
    asymmetric = [.2, -.7, 1.1, .3, -.2, .4, .8, -1.3, .1]
    for theta, x, y in [(INITIAL, (1., 1.), 1), (asymmetric, (-.3, 1.2), 0)]:
        for j, analytic in enumerate(gradient(theta, x, y)):
            plus, minus = theta.copy(), theta.copy()
            plus[j] += 1e-6
            minus[j] -= 1e-6
            numerical = (loss(plus, x, y)-loss(minus, x, y))/2e-6
            assert abs(analytic-numerical) < 1e-8, (j, analytic, numerical)
    orbit = [symmetry(asymmetric, swap, signs)
             for swap in (False, True) for signs in product((-1, 1), repeat=2)]
    assert len(set(map(tuple, orbit))) == 8
    for x in [(1, 1), (-.3, 1.2), (0, 0)]:
        assert all(abs(forward(t, x)[0]-forward(asymmetric, x)[0]) < 1e-12 for t in orbit)
    flipped = symmetry(INITIAL, signs=(-1, -1))
    midpoint = [(a+b)/2 for a, b in zip(INITIAL, flipped)]
    assert midpoint == [0.]*9
    assert abs(loss(INITIAL)-loss(flipped)) < 1e-12
    assert abs(loss(midpoint)-log(2)) < 1e-12
    assert loss(midpoint) > (loss(INITIAL)+loss(flipped))/2
    return actual


def text(x, y, value, cls='ml-label', anchor='middle'):
    return f'<text class="{cls}" x="{x:g}" y="{y:g}" text-anchor="{anchor}">{escape(value)}</text>'


def line(x1, y1, x2, y2, cls='ml-axis', extra=''):
    return f'<line class="{cls}" x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" {extra}/>'


def path(points, cls='ml-curve', extra=''):
    d = ' '.join(('M' if i == 0 else 'L')+f'{x:.2f},{y:.2f}' for i, (x, y) in enumerate(points))
    return f'<path class="{cls}" d="{d}" {extra}/>'


def dot(x, y, r=4, cls='ml-dot'):
    return f'<circle class="{cls}" cx="{x:g}" cy="{y:g}" r="{r:g}"/>'


def save(name, width, height, title, desc, body):
    uid = 'ml-'+name
    marker = (f'<defs><marker id="{uid}-arrow" markerWidth="7" markerHeight="7" '
              'refX="6" refY="3.5" orient="auto-start-reverse"><path d="M0,0 L7,3.5 L0,7" '
              'class="ml-arrowhead"/></marker></defs>')
    body = body.replace('ARROW', f'url(#{uid}-arrow)')
    content = (f'<svg class="ml-diagram" viewBox="0 0 {width} {height}" role="img" '
               f'aria-labelledby="{uid}-title {uid}-desc" xmlns="http://www.w3.org/2000/svg">'
               f'<title id="{uid}-title">{escape(title)}</title><desc id="{uid}-desc">{escape(desc)}</desc>'
               f'{marker}{body}</svg>\n')
    (FIGURES/f'{uid}.svg').write_text(content)


def axes(left, top, width, height, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel):
    xy = lambda x, y: (left+(x-xmin)/(xmax-xmin)*width, top+height-(y-ymin)/(ymax-ymin)*height)
    body = text(left, top-15, ylabel, 'ml-small', 'start')
    for x in xticks:
        a, b = xy(x, ymin)
        body += line(a, top, a, top+height, 'ml-grid')+text(a, b+22, f'{x:g}', 'ml-small')
    for y in yticks:
        a, b = xy(xmin, y)
        body += line(left, b, left+width, b, 'ml-grid')+text(a-10, b+5, f'{y:g}', 'ml-small', 'end')
    body += line(left, top+height, left+width, top+height)+line(left, top, left, top+height)
    body += text(left+width/2, top+height+48, xlabel, 'ml-small')
    return xy, body


def feasibility():
    xy, body = axes(60, 55, 350, 240, 0, 12, 0, 12, [0, 3, 6, 9, 12], [0, 3, 6, 9, 12], 'Distance d (km)', 'Average gradient g (%)')
    body += path([xy(d, 30/d) for d in [2.5+i*9.5/120 for i in range(121)]])
    body += path([xy(d, 10-d/1.2) for d in (0, 12)], 'ml-secondary ml-dashed')
    for d, g, success in [(1.5, 4, 1), (2, 9, 1), (4, 3, 1), (5, 4.5, 1), (7, 2, 1), (9, 2.8, 1),
                          (4, 9, 0), (6, 7, 0), (8, 5.5, 0), (10, 8, 0), (11, 4, 0), (2.8, 11.4, 0)]:
        a, b = xy(d, g)
        if success:
            body += dot(a, b, 4.5)
        else:
            body += line(a-4, b-4, a+4, b+4, 'ml-secondary')+line(a-4, b+4, a+4, b-4, 'ml-secondary')
    body += dot(65, 370)+text(78, 375, 'Completed', 'ml-small', 'start')
    body += line(225, 366, 233, 374, 'ml-secondary')+line(225, 374, 233, 366, 'ml-secondary')
    body += text(242, 375, 'Stopped', 'ml-small', 'start')
    body += text(293, 71, 'dg = C', 'ml-accent-label')+text(302, 93, 'Curved boundary', 'ml-small')
    save('feasibility', 440, 395, 'A nonlinear feasibility boundary', 'Illustrative completed and stopped attempts are separated approximately by the hyperbola distance times gradient equals 300 metres of elevation gain. A dashed straight line cannot match this curve.', body)


def network():
    body = text(210, 24, 'STANDARDIZED INPUTS', 'ml-kicker')
    inputs = [(95, 80), (325, 80)]
    hidden = [(95, 245), (325, 245)]
    output = (210, 410)
    for i, (a, b) in enumerate(inputs):
        for j, (c, d) in enumerate(hidden):
            body += line(a, b+25, c, d-30, 'ml-edge', 'marker-end="ARROW"')
    for x, y in hidden:
        body += line(x, y+30, output[0]+(-20 if x < 210 else 20), output[1]-30, 'ml-edge', 'marker-end="ARROW"')
    for i, (x, y) in enumerate(inputs):
        body += dot(x, y, 26, 'ml-node')+text(x, y+6, f'x{chr(0x2081+i)}')
        body += text(x, 42, 'Length' if i == 0 else 'Gradient', 'ml-small')
    for i, (x, y) in enumerate(hidden):
        body += dot(x, y, 31, 'ml-node')+text(x, y+5, f'z{chr(0x2081+i)}')
        body += text(x+(-18 if i == 0 else 18), y+54, f'tanh · bias b{chr(0x2081+i)}', 'ml-small')
    body += text(56, 174, 'w₁₁', 'ml-accent-label')+text(365, 174, 'w₂₂', 'ml-accent-label')
    body += text(150, 120, 'w₂₁', 'ml-accent-label')+text(270, 120, 'w₁₂', 'ml-accent-label')
    body += text(123, 355, 'v₁', 'ml-accent-label')+text(296, 355, 'v₂', 'ml-accent-label')
    body += dot(210, 410, 35, 'ml-node')+text(210, 416, 'σ')
    body += text(210, 468, 'pθ(x) · success probability').replace('pθ(x)', 'p<tspan baseline-shift="sub" font-size="12">θ</tspan>(x)')+text(210, 494, 'Output bias c', 'ml-small')
    save('network', 420, 515, 'A network with nine parameters', 'Two inputs connect through four weights to two tanh hidden units. Two output weights combine those units, and a sigmoid produces a success probability. Each hidden unit and the output have a bias.', body)


def cross_entropy():
    xy, body = axes(57, 55, 353, 240, 0, 1, 0, 3, [0, .1, .5, .9, 1], [0, 1, 2, 3], 'Predicted success probability p', 'Loss ℓ(p, y)')
    ps = [.05+i*.90/180 for i in range(181)]
    body += path([xy(p, -log(p)) for p in ps])
    body += path([xy(p, -log(1-p)) for p in ps], 'ml-secondary ml-dashed')
    for p in (.1, .5, .9):
        body += dot(*xy(p, -log(p)), 4.5)+dot(*xy(p, -log(1-p)), 4.5, 'ml-ring')
    body += line(60, 370, 90, 370, 'ml-curve')+text(101, 375, 'Success, y = 1', 'ml-small', 'start')
    body += line(242, 370, 272, 370, 'ml-secondary ml-dashed')+text(283, 375, 'Failure, y = 0', 'ml-small', 'start')
    save('cross-entropy', 450, 395, 'Binary cross-entropy for success and failure', 'After success, predictions of 0.1, 0.5, and 0.9 have losses about 2.303, 0.693, and 0.105. After failure those losses reverse. Solid and dashed curves distinguish the outcomes.', body)


def spaces():
    xy, body = axes(46, 66, 238, 196, 0, 12, 0, 12, [0, 6, 12], [0, 6, 12], 'Distance (km)', 'Gradient (%)')
    heat = ''
    for i in range(24):
        for j in range(24):
            d, g = (i+.5)/2, (j+.5)/2
            p = 1/(1+exp((d*g-30)/8))
            a, b = xy(i/2, (j+1)/2)
            heat += f'<rect class="ml-heat" x="{a:g}" y="{b:g}" width="10.1" height="8.3" opacity="{.06+.82*p:.3f}"/>'
    body = heat+body
    body += text(165, 23, 'Input space')+text(165, 335, 'Stronger color = higher probability', 'ml-small')
    save('input-space', 320, 355, 'Input space contains climbs', 'A schematic probability field varies over distance and gradient, with higher success probability at smaller products. Parameters are held fixed.', body)
    xy, body = axes(46, 66, 238, 196, -2, 2, -2, 2, [-2, 0, 2], [-2, 0, 2], 'Parameter θ₁', 'Parameter θ₂')
    for r in (.4, .8, 1.2, 1.6):
        a, b = xy(0, 0)
        body += f'<ellipse class="ml-contour" cx="{a:g}" cy="{b:g}" rx="{r*49:g}" ry="{r*30:g}" transform="rotate(-25 {a:g} {b:g})"/>'
    body += dot(*xy(1.3, 1.1), 4.5)+text(205, 93, 'One model', 'ml-small')
    body += text(165, 23, 'Parameter space')+text(165, 335, 'Each contour has equal loss', 'ml-small')
    save('parameter-space', 320, 355, 'Parameter space contains models', 'Schematic contours of equal loss in two parameter directions, with the dataset held fixed. This is not a computed slice of the neural network.', body)


def descent():
    xy, body = axes(44, 67, 230, 185, -2, 2, 0, 2.2, [-2, 0, 2], [0, 1, 2], 'Parameter u', 'q(u) = u² / 2')
    body += path([xy(u, u*u/2) for u in [-2+i/50 for i in range(201)]])
    body += dot(*xy(1, .5), 5)
    body += line(*xy(1, .5), *xy(1.65, 1.15), 'ml-edge', 'marker-end="ARROW"')
    body += line(*xy(1, .5), *xy(.6, .1), 'ml-edge', 'marker-end="ARROW"')
    body += text(242, 122, '∇q', 'ml-accent-label')+text(169, 220, '−∇q', 'ml-accent-label')
    body += text(160, 23, 'Choose a direction')+text(160, 325, 'At u = 1, the derivative is +1', 'ml-small')
    save('descent-direction', 320, 345, 'The negative gradient points downhill', 'For q(u)=u squared over two, the derivative at u=1 is positive one. The positive gradient points uphill and its negative points toward the minimum at zero.', body)
    xy, body = axes(44, 67, 230, 185, 0, 6, -3.2, 3.2, [0, 2, 4, 6], [-3, 0, 3], 'Iteration k', 'Parameter uₖ')
    for eta, cls in [(.5, 'ml-curve'), (2.2, 'ml-secondary ml-dashed')]:
        values = [(k, (1-eta)**k) for k in range(7)]
        body += path([xy(*p) for p in values], cls)
        body += ''.join(dot(*xy(*p), 3, 'ml-dot' if eta < 1 else 'ml-ring') for p in values)
    body += text(160, 23, 'Choose a step size')
    body += line(46, 320, 69, 320, 'ml-curve')+text(78, 325, 'η = 0.5', 'ml-small', 'start')
    body += line(170, 320, 193, 320, 'ml-secondary ml-dashed')+text(202, 325, 'η = 2.2', 'ml-small', 'start')
    save('descent-steps', 320, 345, 'A large learning rate can diverge', 'Computed iterations from u0=1: learning rate 0.5 halves the parameter at each step, while learning rate 2.2 alternates sign and multiplies its magnitude by 1.2.', body)


def numerical():
    p, ell, pn, elln = verify_example()
    body = text(220, 27, 'ONE SUCCESSFUL ATTEMPT', 'ml-kicker')
    for x, heading, a, b in [(112, 'Before', p, ell), (328, 'After', pn, elln)]:
        body += f'<rect class="ml-card" x="{x-91}" y="55" width="182" height="190" rx="3"/>'
        body += text(x, 86, heading, 'ml-small')+text(x, 125, f'{a:.4f}', 'ml-number')
        body += text(x, 148, 'Success probability', 'ml-small')
        body += text(x, 192, f'{b:.4f}', 'ml-number')+text(x, 216, 'Cross-entropy', 'ml-small')
    body += line(205, 151, 230, 151, 'ml-edge', 'marker-end="ARROW"')
    body += text(220, 280, 'θnew = θ − 0.1 ∇ℓ(θ)', 'ml-accent-label').replace('θnew', 'θ<tspan baseline-shift="sub" font-size="12">new</tspan>')
    save('numerical-step', 440, 305, 'One gradient step reduces the example loss', 'The probability increases from 0.7159 to 0.7443 and the cross-entropy decreases from 0.3342 to 0.2954 after one simultaneous parameter update with learning rate 0.1.', body)


def landscape():
    # Critical points satisfy 4u³−4u+0.3=0. Bisection avoids optional dependencies.
    def root(a, b):
        f = lambda u: 4*u**3-4*u+.3
        for _ in range(70):
            m = (a+b)/2
            if f(a)*f(m) <= 0:
                b = m
            else:
                a = m
        return (a+b)/2
    minima = [root(-1.3, -.8), root(.8, 1.2)]
    saddle = root(0, .2)
    f = lambda u: (u*u-1)**2+.3*u
    # A true 2-D contour plot: v=±sqrt(level-f(u)).
    xy, body = axes(55, 60, 365, 230, -1.75, 1.75, -1.35, 1.35, [-1, 0, 1], [-1, 0, 1], 'Parameter direction u', 'Parameter direction v')
    for level in [-.2, 0, .4, .7, 1., 1.25, 1.6, 2.]:
        for sign in (-1, 1):
            segment = []
            for i in range(701):
                u = -1.75+3.5*i/700
                delta = level-f(u)
                if delta >= 0 and sqrt(delta) <= 1.35:
                    segment.append(xy(u, sign*sqrt(delta)))
                else:
                    if len(segment) > 1:
                        body += path(segment, 'ml-contour')
                    segment = []
            if len(segment) > 1:
                body += path(segment, 'ml-contour')
    for start in [(-1.45, 1.), (1.35, -.9)]:
        u, v = start
        points = [xy(u, v)]
        for _ in range(35):
            u -= .035*(4*u*(u*u-1)+.3)
            v -= .035*2*v
            points.append(xy(u, v))
        body += path(points, 'ml-curve', 'marker-end="ARROW"')+dot(*points[0], 3)
    for u in minima:
        body += dot(*xy(u, 0), 5)
    body += dot(*xy(saddle, 0), 5, 'ml-ring')
    body += text(117, 215, 'Lower minimum', 'ml-small')
    body += text(344, 132, 'Higher minimum', 'ml-small')
    body += text(250, 157, 'Saddle', 'ml-small')
    body += text(240, 24, 'SCHEMATIC LANDSCAPE', 'ml-kicker')
    body += text(240, 368, 'Different minima ≠ equivalent parameter copies', 'ml-small')
    save('local-minima', 460, 390, 'Schematic landscape with two unequal minima and a saddle', 'Contours of F(u,v)=(u squared minus one) squared plus 0.3u plus v squared. The left minimum is lower. Gradient descent from two different starting regions follows different valleys. The stationary point between the minima is a saddle, not a minimum. This is not the network loss.', body)


def flow():
    for name, title, labels, notes in [
        ('training', 'Training', ['Recorded attempts', 'Model → predictions', 'Loss against outcomes', 'Backpropagation', 'Update parameters'], ['Inputs x and outcomes y', 'Current parameters θ', 'Measure the discrepancy', 'Compute the gradient', 'Gradient descent']),
        ('prediction', 'Prediction', ['New climb', 'Trained model', 'Success probability'], ['Input x', 'Parameters held fixed', 'No parameter update'])]:
        body = text(155, 25, title)
        for i, (label, note) in enumerate(zip(labels, notes)):
            y = 55+i*81
            body += f'<rect class="ml-card" x="35" y="{y}" width="240" height="58" rx="3"/>'
            body += text(155, y+24, label, 'ml-flow-label')+text(155, y+44, note, 'ml-small')
            if i < len(labels)-1:
                body += line(155, y+60, 155, y+77, 'ml-edge', 'marker-end="ARROW"')
        if name == 'training':
            body += '<path class="ml-edge" d="M275,408 L299,408 L299,165 L278,165" marker-end="ARROW"/>'
            body += text(155, 476, 'Repeat with updated θ', 'ml-small')
        else:
            body += text(155, 368, 'Use what has been learned.', 'ml-label')
            body += text(155, 393, 'Keep the parameters fixed.', 'ml-small')
        save(name, 310, 495, title+' process', 'Recorded examples drive a loop of predictions, loss, backpropagation, and parameter updates.' if name == 'training' else 'A new climb passes through the trained model to produce a success probability, with no parameter update.', body)


if __name__ == '__main__':
    verify_example()
    FIGURES.mkdir(exist_ok=True)
    feasibility()
    network()
    cross_entropy()
    spaces()
    descent()
    numerical()
    landscape()
    flow()
    print('Generated 11 theme-aware SVGs; verified numerical update, finite-difference gradients, and eight network symmetries.')
